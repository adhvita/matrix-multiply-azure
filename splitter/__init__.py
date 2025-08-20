import os, io, json, uuid, logging
import azure.functions as func
import numpy as np
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient

# ------- Tunables -------
DEFAULT_BLOCK           = int(os.getenv("TILE_BLOCK", "96"))      # tile size for A,B
DEFAULT_TILES_PER_SHARD = int(os.getenv("TILES_PER_SHARD", "1"))  # (i,j) tiles per shard

TEMP_CONTAINER = os.getenv("TEMP_CONTAINER", "temp")
PROCESS_QUEUE  = os.getenv("PROCESS_QUEUE",  "process-queue")

# ----------------- Clients -----------------
def _bsc() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _qc():
    qc = QueueClient.from_connection_string(os.environ["AzureWebJobsStorage"], PROCESS_QUEUE)
    try:
        qc.create_queue()
    except ResourceExistsError:
        pass                      # benign race: queue already exists
    except Exception as e:
        logging.warning(f"Queue create check: {e}")  # don’t crash the function
    return qc

# ----------------- Utils -----------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _send_shard_msg(qc: QueueClient, run_id: str, shard_no: int, shard_path: str):
    qc.send_message(json.dumps({
        "run_id": run_id,
        "shard_no": shard_no,
        "shard_blob": f"{TEMP_CONTAINER}/{shard_path}"
    }))

def _upload_bytes(container: str, name: str, data: bytes, content_type: str):
    _bsc().get_blob_client(container, name).upload_blob(
        data, overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )

# ----------------- Binary shard writer -----------------
def _write_binary_shard(run_id: str, shard_idx: int, tasks: list) -> str:
    """
    tasks: list of dicts with keys:
      i0,j0,k0,bi,bj,bk and Ablk (np.ndarray float32), Bblk (np.ndarray float32)
    Produces a single .npz containing multiple tasks:
      scalars/1D: N,i0,j0,k0,bi,bj,bk
      per-task 2D arrays: A_000,B_000,...
    """
    N = len(tasks)
    i0 = np.fromiter((t["i0"] for t in tasks), count=N, dtype=np.int32)
    j0 = np.fromiter((t["j0"] for t in tasks), count=N, dtype=np.int32)
    k0 = np.fromiter((t["k0"] for t in tasks), count=N, dtype=np.int32)
    bi = np.fromiter((t["bi"] for t in tasks), count=N, dtype=np.int32)
    bj = np.fromiter((t["bj"] for t in tasks), count=N, dtype=np.int32)
    bk = np.fromiter((t["bk"] for t in tasks), count=N, dtype=np.int32)

    payload = {"N": np.array(N, dtype=np.int64), "i0": i0, "j0": j0, "k0": k0,
               "bi": bi, "bj": bj, "bk": bk}
    for idx, t in enumerate(tasks):
        payload[f"A_{idx:03d}"] = t["Ablk"].astype(np.float32, copy=False)
        payload[f"B_{idx:03d}"] = t["Bblk"].astype(np.float32, copy=False)

    buf = io.BytesIO()
    np.savez(buf, **payload)
    buf.seek(0)

    shard_name = f"runs/{run_id}/shards/shard-{shard_idx:05d}.npz"
    _upload_bytes(TEMP_CONTAINER, shard_name, buf.getvalue(), "application/octet-stream")
    return shard_name

# ----------------- Loader for .npy input -----------------
def _extract_AB_from_npy_bytes(data: bytes):
    """
    Accepts a .npy saved with allow_pickle=True containing:
      - dict with keys 'A' and 'B', or
      - tuple/list (A, B)
    """
    arr = np.load(io.BytesIO(data), allow_pickle=True)
    obj = arr
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()

    if isinstance(obj, dict) and "A" in obj and "B" in obj:
        return obj["A"], obj["B"]
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return obj[0], obj[1]
    raise ValueError("Input .npy must contain dict {A,B} or tuple/list (A,B)")

# ----------------- Main -----------------
def main(inBlob: func.InputStream, queueOut: func.Out[str]):
    base = os.path.splitext(os.path.basename(inBlob.name))[0]
    run_id = f"{base}-{uuid.uuid4().hex[:8]}"

    in_container, in_name = inBlob.name.split("/", 1)
    bc_in = _bsc().get_blob_client(in_container, in_name)
    name_lower = in_name.lower()

    if not name_lower.endswith(".npy"):
        raise ValueError("Only .npy inputs are supported (dict {A,B} or tuple (A,B))")

    data = bc_in.download_blob(max_concurrency=2).readall()
    A, B = _extract_AB_from_npy_bytes(data)
    block = DEFAULT_BLOCK
    tiles_per_shard = DEFAULT_TILES_PER_SHARD
    mode = "tile-binary-input-npy"

    # --- Validate ---
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("A and B must be numpy arrays")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D; got A.ndim={A.ndim}, B.ndim={B.ndim}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shape mismatch: A {A.shape}, B {B.shape}")
    if A.dtype != np.float32: A = A.astype(np.float32, copy=False)
    if B.dtype != np.float32: B = B.astype(np.float32, copy=False)

    m, n = A.shape
    _, p = B.shape

    qc = _qc()
    shard_idx, tasks, tiles_in_shard = 0, [], 0

    def flush():
        nonlocal shard_idx, tasks, tiles_in_shard
        if not tasks: return
        shard_idx += 1
        shard_blob = _write_binary_shard(run_id, shard_idx, tasks)
        _send_shard_msg(qc, run_id, shard_idx, shard_blob)
        logging.info("Shard %d written (.npz): tasks=%d", shard_idx, len(tasks))
        tasks.clear(); tiles_in_shard = 0

    for i0 in range(0, m, block):
        bi = min(block, m - i0)
        for j0 in range(0, p, block):
            bj = min(block, p - j0)
            if tiles_in_shard >= tiles_per_shard and tasks:
                flush()
            for k0 in range(0, n, block):
                bk = min(block, n - k0)
                Ablk = A[i0:i0+bi, k0:k0+bk]
                Bblk = B[k0:k0+bk, j0:j0+bj]
                tasks.append({"i0": int(i0), "j0": int(j0), "k0": int(k0),
                              "bi": int(bi), "bj": int(bj), "bk": int(bk),
                              "Ablk": Ablk, "Bblk": Bblk})
            tiles_in_shard += 1
    flush()

    meta = {
        "run_id": run_id,
        "shapeA": [int(m), int(n)],
        "shapeB": [int(n), int(p)],
        "block": int(block),
        "tiles_per_shard": int(tiles_per_shard),
        "num_shards": int(shard_idx),
        "created_utc": _now_iso(),
        "mode": mode
    }
    _upload_bytes(TEMP_CONTAINER, f"runs/{run_id}/meta.json",
                  json.dumps(meta, indent=2).encode("utf-8"), "application/json")
    manifest = {
        "run_id": run_id,
        "expected_parts": int(shard_idx),
        "created_utc": _now_iso(),
        "mode": mode
    }
    _upload_bytes(TEMP_CONTAINER, f"runs/{run_id}/manifest.json",
                  json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    logging.info("Split complete: run=%s mode=%s shards=%d", run_id, mode, shard_idx)
