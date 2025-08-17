import os, io, json, uuid, logging
import azure.functions as func
import ijson
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient

# ------- Tunables -------
DEFAULT_MAX_LINES   = int(os.getenv("SHARD_MAX_LINES", "400"))                 # rows per shard (pairs mode)
DEFAULT_MAX_BYTES   = int(os.getenv("SHARD_MAX_BYTES", str(6 * 1024 * 1024)))  # ~6 MiB safety cap
DEFAULT_COST_BUDGET = float(os.getenv("SHARD_COST_BUDGET", "2.0e7"))           # rough work cap (pairs mode)
DEFAULT_BLOCK       = int(os.getenv("TILE_BLOCK", "128"))                      # block size (tile mode)
DEFAULT_TILES_PER_SHARD = int(os.getenv("TILES_PER_SHARD", "2"))               # (i,j) tiles per shard (tile mode)

READ_BUF_MB    = 16
TEMP_CONTAINER = os.getenv("TEMP_CONTAINER", "temp")
PROCESS_QUEUE  = os.getenv("PROCESS_QUEUE",  "process-queue")

def _bsc():
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _qc():
    qc = QueueClient.from_connection_string(os.environ["AzureWebJobsStorage"], PROCESS_QUEUE)
    qc.create_queue()
    return qc

def _iter_text(bc):
    """Stream a blob as text with bounded memory."""
    chunks = bc.download_blob(max_concurrency=2).chunks()
    class _Raw(io.RawIOBase):
        def __init__(self, it): self.it, self.buf = iter(it), b""
        def readable(self): return True
        def readinto(self, b):
            if not self.buf:
                try: self.buf = next(self.it)
                except StopIteration: return 0
            n = min(len(b), len(self.buf))
            b[:n] = self.buf[:n]
            self.buf = self.buf[n:]
            return n
    return io.TextIOWrapper(
        io.BufferedReader(_Raw(chunks), READ_BUF_MB * 1024 * 1024),
        encoding="utf-8"
    )

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def estimate_cost(pair) -> float:
    """Cheap ~n^2.807 proxy so we keep per-shard work bounded."""
    try:
        A = pair.get("A")
        if isinstance(A, list) and A:
            n = min(len(A), len(A[0]) if isinstance(A[0], list) and A[0] else len(A))
        else:
            n = 64
        if n <= 0: n = 64
    except Exception:
        n = 64
    return float(n ** 2.807)

def _write_blob(path: str, data: bytes, content_type: str):
    _bsc().get_blob_client(TEMP_CONTAINER, path).upload_blob(
        data, overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )

def _send_shard_msg(qc: QueueClient, run_id: str, shard_no: int, shard_path: str):
    qc.send_message(json.dumps({
        "run_id": run_id,
        "shard_no": shard_no,
        "shard_blob": f"{TEMP_CONTAINER}/{shard_path}"
    }))

# ---------- TILE MODE ----------
def _emit_tiles_from_single_object(obj: dict, run_id: str, block: int, tiles_per_shard: int) -> int:
    """Emit shards from a single large A×B object."""
    A, B = obj["A"], obj["B"]
    m, n = len(A), len(A[0]) if A else 0
    n2, p = len(B), len(B[0]) if B else 0
    if n != n2:
        raise ValueError(f"shape mismatch: A {m}x{n}, B {n2}x{p}")

    qc = _qc()
    shard_idx = 0
    lines, bytes_used, tiles_in_shard = [], 0, 0
    prefix = f"runs/{run_id}/shards"

    def flush():
        nonlocal shard_idx, lines, bytes_used, tiles_in_shard
        if not lines: return
        shard_idx += 1
        shard_name = f"{prefix}/shard-{shard_idx:05d}.ndjson"
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        _write_blob(shard_name, payload, "application/x-ndjson")
        _send_shard_msg(qc, run_id, shard_idx, shard_name)
        logging.info("Shard %d written (tile mode): %d rows, %.2f MiB",
                     shard_idx, len(lines), len(payload)/(1024*1024))
        lines.clear(); bytes_used = 0; tiles_in_shard = 0

    for i0 in range(0, m, block):
        bi = min(block, m - i0)
        for j0 in range(0, p, block):
            bj = min(block, p - j0)
            if tiles_in_shard >= tiles_per_shard and lines:
                flush()
            for k0 in range(0, n, block):
                bk = min(block, n - k0)
                Ablk = [row[k0:k0+bk] for row in A[i0:i0+bi]]
                Bblk = [row[j0:j0+bj] for row in B[k0:k0+bk]]
                line = json.dumps({
                    "run_id": run_id, "i0": i0, "j0": j0, "k0": k0,
                    "bi": bi, "bj": bj, "bk": bk, "A": Ablk, "B": Bblk
                }, separators=(",", ":"))
                if bytes_used + len(line) + 1 > DEFAULT_MAX_BYTES:
                    flush()
                lines.append(line); bytes_used += len(line) + 1
            tiles_in_shard += 1
    flush()
    return shard_idx, (m, n, p)

# ---------- MAIN ----------
def main(inBlob: func.InputStream, queueOut: func.Out[str]):  # queueOut unused (we use QueueClient)
    base = os.path.splitext(os.path.basename(inBlob.name))[0]
    run_id = f"{base}-{uuid.uuid4().hex[:8]}"

    bc_in = _bsc().get_blob_client(*inBlob.name.split("/", 1))
    raw = bc_in.download_blob(max_concurrency=2).readall().decode("utf-8")
    raw_stripped = raw.lstrip()

    shard_idx, total_pairs, mode, shapes = 0, 0, None, None

    if raw_stripped.startswith("{"):
        # ---- TILE MODE ----
        obj = json.loads(raw)
        block = int(obj.get("block", DEFAULT_BLOCK))
        tiles_per_shard = int(obj.get("tiles_per_shard", DEFAULT_TILES_PER_SHARD))
        shard_idx, (m, n, p) = _emit_tiles_from_single_object(obj, run_id, block, tiles_per_shard)
        mode, shapes = "tile", (m, n, p)
        total_pairs = ((m + block - 1)//block) * ((p + block - 1)//block)

        meta = {
            "run_id": run_id,
            "shapeA": [int(m), int(n)],
            "shapeB": [int(n), int(p)],
            "block": block,
            "num_shards": shard_idx,
            "created_utc": _now_iso(),
            "mode": mode
        }
        _write_blob(f"runs/{run_id}/meta.json", json.dumps(meta, indent=2).encode(), "application/json")

    elif raw_stripped.startswith("["):
        # ---- ARRAY-OF-PAIRS MODE ----
        mode = "pairs"
        qc = _qc()
        prefix = f"runs/{run_id}/shards"
        buf_lines, buf_bytes, cost_used = [], 0, 0.0

        def flush():
            nonlocal shard_idx, buf_lines, buf_bytes, cost_used
            if not buf_lines: return
            shard_idx += 1
            shard_name = f"{prefix}/shard-{shard_idx:05d}.ndjson"
            data = ("\n".join(buf_lines) + "\n").encode("utf-8")
            _write_blob(shard_name, data, "application/x-ndjson")
            _send_shard_msg(qc, run_id, shard_idx, shard_name)
            logging.info("Shard %d written: %d rows, %.2f MiB",
                         shard_idx, len(buf_lines), len(data)/(1024*1024))
            buf_lines.clear(); buf_bytes = 0; cost_used = 0.0

        txt_iter = _iter_text(bc_in)
        for pair in ijson.items(txt_iter, "item"):
            line = json.dumps(pair, separators=(",", ":"))
            size = len(line) + 1
            cst = estimate_cost(pair)
            if buf_lines and (len(buf_lines)+1 > DEFAULT_MAX_LINES or
                              buf_bytes+size > DEFAULT_MAX_BYTES or
                              cost_used+cst > DEFAULT_COST_BUDGET):
                flush()
            buf_lines.append(line)
            buf_bytes += size
            cost_used += cst
            total_pairs += 1
            if len(buf_lines) >= DEFAULT_MAX_LINES or buf_bytes >= DEFAULT_MAX_BYTES or cost_used >= DEFAULT_COST_BUDGET:
                flush()
        flush()

        meta = {
            "run_id": run_id,
            "num_shards": shard_idx,
            "created_utc": _now_iso(),
            "mode": mode
        }
        _write_blob(f"runs/{run_id}/meta.json", json.dumps(meta, indent=2).encode(), "application/json")

    else:
        raise ValueError("Input must be a JSON object ({A,B}) or an array of {A,B} pairs")

    manifest = {
        "run_id": run_id,
        "expected_parts": shard_idx,
        "created_utc": _now_iso(),
        "max_lines": DEFAULT_MAX_LINES,
        "max_bytes": DEFAULT_MAX_BYTES,
        "cost_budget": DEFAULT_COST_BUDGET,
        "total_pairs": total_pairs,
        "mode": mode
    }
    _write_blob(f"runs/{run_id}/manifest.json", json.dumps(manifest, indent=2).encode(), "application/json")

    logging.info("Split complete: run=%s mode=%s shards=%d total_pairs=%d",
                 run_id, mode, shard_idx, total_pairs)
