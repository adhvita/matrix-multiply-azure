from __future__ import annotations

import os
import io
import json
import time
import base64
import logging
from typing import Dict, Tuple
from datetime import datetime, timezone

import numpy as np
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient
from azure.core.exceptions import ResourceNotFoundError

from strassen_algo.strassen_module import strassen  # your algo


logging.warning("process_chunk: MODULE LOADED build=%s",
                os.getenv("WEBSITE_INSTANCE_ID", "local"))

# ---- Settings / constants ----
TEMP_CONTAINER = os.getenv("TEMP_CONTAINER", "temp")
PROCESS_QUEUE  = os.getenv("PROCESS_QUEUE",  "process-queue")

# Time budget (Consumption-safe)
MAX_SECONDS   = int(os.getenv("CHUNK_TIME_BUDGET_SEC", "240"))   # ~4 minutes
SAFETY_MARGIN = int(os.getenv("CHUNK_SAFETY_MARGIN_SEC", "15"))  # stop with margin

def _cs() -> str:
    cs = os.environ.get("AzureWebJobsStorage")
    if not cs:
        raise RuntimeError("AzureWebJobsStorage not set")
    return cs

def _cs_queue() -> str:  # <<< CHANGED
    return os.environ.get("QueueConnection") or _cs()  # <<< CHANGED

def _bsc() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(_cs())

def _qc() -> QueueClient:
    return QueueClient.from_connection_string(_cs_queue(), PROCESS_QUEUE)  # <<< CHANGED

def _upload_bytes(container: str, name: str, data: bytes, content_type: str):
    _bsc().get_blob_client(container, name).upload_blob(
        data, overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )

def _write_binary_shard(run_id: str, shard_no: int, tasks: list) -> str:
    """
    Write a continuation shard in the same format as the splitter:
      scalars/1D: N,i0,j0,k0,bi,bj,bk
      per-task 2D arrays: A_000,B_000, A_001,B_001, ...
    """
    N  = len(tasks)
    i0 = np.fromiter((t["i0"] for t in tasks), count=N, dtype=np.int32)
    j0 = np.fromiter((t["j0"] for t in tasks), count=N, dtype=np.int32)
    k0 = np.fromiter((t["k0"] for t in tasks), count=N, dtype=np.int32)
    bi = np.fromiter((t["bi"] for t in tasks), count=N, dtype=np.int32)
    bj = np.fromiter((t["bj"] for t in tasks), count=N, dtype=np.int32)
    bk = np.fromiter((t["bk"] for t in tasks), count=N, dtype=np.int32)

    payload = {"N": np.array(N, dtype=np.int64), "i0": i0, "j0": j0, "k0": k0, "bi": bi, "bj": bj, "bk": bk}
    for idx, t in enumerate(tasks):
        payload[f"A_{idx:03d}"] = t["Ablk"].astype(np.float32, copy=False)
        payload[f"B_{idx:03d}"] = t["Bblk"].astype(np.float32, copy=False)

    buf = io.BytesIO()
    np.savez(buf, **payload)
    buf.seek(0)

    # Use a timestamp to make the blob name unique per continuation
    cont_name = f"runs/{run_id}/shards/shard-{shard_no:05d}-cont-{int(time.time())}.npz"
    _upload_bytes(TEMP_CONTAINER, cont_name, buf.getvalue(), "application/octet-stream")
    return cont_name  # path only (container is TEMP_CONTAINER)

def _write_part_npz(run_id: str, shard_no: int, tiles: Dict[Tuple[int,int], np.ndarray] | None):
    """
    tiles: {(i0,j0): C_tile ndarray(float32)}
    Writes/overwrites temp/runs/<run_id>/parts/part-xxxxx.npz
    """
    if not tiles:
        return None

    items = sorted(tiles.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    M   = len(items)
    i0s = np.fromiter((k[0] for k, _ in items), count=M, dtype=np.int32)
    j0s = np.fromiter((k[1] for k, _ in items), count=M, dtype=np.int32)
    bis = np.fromiter((v.shape[0] for _, v in items), count=M, dtype=np.int32)
    bjs = np.fromiter((v.shape[1] for _, v in items), count=M, dtype=np.int32)

    payload = {"M": np.array(M, dtype=np.int64), "i0": i0s, "j0": j0s, "bi": bis, "bj": bjs}
    for idx, (_, C) in enumerate(items):
        payload[f"C_{idx:03d}"] = C.astype(np.float32, copy=False)

    buf = io.BytesIO()
    np.savez(buf, **payload)
    buf.seek(0)

    part_name = f"runs/{run_id}/parts/part-{shard_no:05d}.npz"
    _upload_bytes(TEMP_CONTAINER, part_name, buf.getvalue(), "application/octet-stream")
    return part_name

#def _parse_queue_payload(msg: func.QueueMessage) -> dict:
    raw = msg.get_body()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    # try plain UTF-8 JSON
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass
    # try base64 -> JSON
    try:
        return json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception:
        logging.error("process_chunk: cannot parse queue payload. First 200B=%r", raw[:200])
        raise
def _parse_queue_payload(msg: func.QueueMessage) -> dict:
    import base64, json, logging

    raw = msg.get_body()            # may be bytes or str depending on host
    if raw is None:
        raise ValueError("empty queue message")

    # Normalize to str
    if isinstance(raw, (bytes, bytearray)):
        s = raw.decode("utf-8", errors="ignore")
    else:
        s = str(raw)
    s = s.strip()
    if not s:
        raise ValueError("empty queue message string")

    # 1) Try plain JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Try base64 -> JSON
    try:
        s2 = base64.b64decode(s).decode("utf-8")
        obj = json.loads(s2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) Sometimes message is a JSON *string* of JSON (double-encoded)
    try:
        if s.startswith('"') and s.endswith('"'):
            obj = json.loads(json.loads(s))
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass

    logging.error("process_chunk: cannot parse queue payload. First 200B=%r", s[:200])
    raise ValueError("Unparsable queue payload")

def _next_pow2(x: int) -> int:  # <<< CHANGED
    return 1 << (x - 1).bit_length()  # handles x>=1

STRASSEN_MIN = int(os.getenv("STRASSEN_MIN", "64"))  # <<< CHANGED (tunable cutoff)

def strassen_matmul_safe(A: np.ndarray, B: np.ndarray) -> np.ndarray:  # <<< CHANGED
    assert A.ndim == 2 and B.ndim == 2, f"Need 2D, got A.ndim={A.ndim}, B.ndim={B.ndim}"
    m, k1 = A.shape
    k2, n = B.shape
    if k1 != k2:
        raise ValueError(f"Incompatible inner dims: A{A.shape} @ B{B.shape}")

    # Small/skinny tiles → classical is faster/safer
    if max(m, n, k1) < STRASSEN_MIN:
        return A @ B

    # Already square & same size → straight Strassen
    if A.shape == B.shape and m == k1:
        return strassen(A, B)

    # Pad to S×S (power of two)
    S = _next_pow2(max(m, k1, n))
    Ap = np.zeros((S, S), dtype=A.dtype);  Ap[:m, :k1] = A
    Bp = np.zeros((S, S), dtype=B.dtype);  Bp[:k2, :n] = B
    Cp = strassen(Ap, Bp)
    return Cp[:m, :n]

def main(msg: func.QueueMessage, mergeOut: func.Out[str]) -> None:
    try:
        payload    = _parse_queue_payload(msg)
        run_id     = payload["run_id"]
        shard_no   = int(payload["shard_no"])
        shard_blob = payload["shard_blob"]  # e.g. "temp/runs/<run_id>/shards/shard-00001.npz"

        logging.info("process_chunk start run=%s shard=%d shard_blob=%s", run_id, shard_no, shard_blob)

        # Resolve blob
        if "/" in shard_blob:
            container, name = shard_blob.split("/", 1)
        else:
            container, name = TEMP_CONTAINER, shard_blob

        bsc  = _bsc()
        qc   = _qc()
        bc_in = bsc.get_blob_client(container=container, blob=name)

        # A) Does the blob exist?
        if not bc_in.exists():
            logging.error("process_chunk: shard blob NOT FOUND: %s/%s", container, name)
            raise RuntimeError("Shard blob missing")

        # B) Download and open .npz
        try:
            data = bc_in.download_blob(max_concurrency=2).readall()
        except ResourceNotFoundError:
            logging.error("process_chunk: 404 while downloading shard %s/%s", container, name)
            raise

        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            files = set(z.files)
            required = {"N", "i0", "j0", "k0", "bi", "bj", "bk"}
            if not required.issubset(files):
                logging.error("process_chunk: bad shard format; required=%s have=%s", required, files)
                raise ValueError("Bad shard format")

            N = int(z["N"])
            if N <= 0:
                logging.error("process_chunk: N<=0 in shard")
                raise ValueError("Empty shard")

            started = time.monotonic()
            def time_left():
                return MAX_SECONDS - (time.monotonic() - started)

            tiles: Dict[Tuple[int,int], np.ndarray] = {}
            next_idx = 0

            # Process tasks within the time budget
            while next_idx < N:
                if time_left() < SAFETY_MARGIN:
                    break

                i0 = int(z["i0"][next_idx]); j0 = int(z["j0"][next_idx])
                A  = z[f"A_{next_idx:03d}"]
                B  = z[f"B_{next_idx:03d}"]

                
                if A.ndim != 2 or B.ndim != 2:
                    logging.error("process_chunk: tile %d not 2D: A.ndim=%d B.ndim=%d", next_idx, A.ndim, B.ndim)
                    raise ValueError("Tile not 2D")
                if A.shape[1] != B.shape[0]:
                    logging.error("process_chunk: tile %d shape mismatch A%s x B%s", next_idx, A.shape, B.shape)
                    raise ValueError(f"Incompatible tile shapes: A{A.shape} @ B{B.shape}")

                logging.info("process_chunk: tile %d A%s @ B%s", next_idx, A.shape, B.shape)  # <<< CHANGED

                C_part = strassen_matmul_safe(A, B)  # <<< CHANGED

                key = (i0, j0)
                tiles[key] = C_part if key not in tiles else tiles[key] + C_part
                next_idx += 1

        # Write partial/final tiles for this invocation
        part_blob = _write_part_npz(run_id, shard_no, tiles)

        # If tasks remain, emit a continuation shard and re-enqueue (same shard_blob format)
        if next_idx < N:
            remaining = []
            with np.load(io.BytesIO(data), allow_pickle=False) as z2:
                for idx in range(next_idx, N):
                    remaining.append({
                        "i0": int(z2["i0"][idx]),
                        "j0": int(z2["j0"][idx]),
                        "k0": int(z2["k0"][idx]),
                        "bi": int(z2["bi"][idx]),
                        "bj": int(z2["bj"][idx]),
                        "bk": int(z2["bk"][idx]),
                        "Ablk": z2[f"A_{idx:03d}"],
                        "Bblk": z2[f"B_{idx:03d}"]
                    })
            cont_path = _write_binary_shard(run_id, shard_no, remaining)
            qc.send_message(json.dumps({
                "run_id": run_id,
                "shard_no": shard_no,                                
                "shard_blob": f"{TEMP_CONTAINER}/{cont_path}"        
            }))
            logging.info("Continuation shard queued: %s/%s", TEMP_CONTAINER, cont_path)
        else:
            # No tasks left: mark this original shard as DONE
            done_name = f"runs/{run_id}/parts/part-{shard_no:05d}.done"
            _upload_bytes(TEMP_CONTAINER, done_name, b"ok", "text/plain")

        logging.info("process_chunk done run=%s shard=%d tiles_written=%d",
                     run_id, shard_no, 0 if not tiles else len(tiles))

        # Notify merger (idempotent; merger will wait for all .done files)
        mergeOut.set(json.dumps({
            "run_id": run_id,
            "shard_no": shard_no,
            "part_blob": f"{TEMP_CONTAINER}/{part_blob}" if part_blob else None,
            "ts": datetime.now(timezone.utc).isoformat()
        }))

    except Exception:
        logging.exception("process_chunk failed")
        raise
