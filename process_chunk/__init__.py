import os, io, json, time, logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobType, ContentSettings
from azure.core.exceptions import ResourceExistsError
import ijson, numpy as np

# ⬅️ adjust if your module path differs
from ..strassen_algo.strassen_module import strassen

MAX_RUN_SECONDS = 120     # keep runs short; pipeline resumes via queue
JSONL_PREVIEW   = 8

# ---------- Strassen with power-of-two padding ----------
def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def _pad_to_pow2(M: np.ndarray, size: int) -> np.ndarray:
    if M.shape[0] == size:
        return M
    P = np.zeros((size, size), dtype=M.dtype)
    P[: M.shape[0], : M.shape[1]] = M
    return P

def _strassen_padded(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    m = _next_pow2(n)
    Ap = _pad_to_pow2(A, m)
    Bp = _pad_to_pow2(B, m)
    Cp = strassen(Ap, Bp)      # ALWAYS use Strassen
    return Cp[:n, :n]

# ---------- Streaming input ----------
class _IterStream(io.RawIOBase):
    def __init__(self, it): self.it, self.buf = iter(it), b""
    def readable(self): return True
    def readinto(self, b):
        if not self.buf:
            try: self.buf = next(self.it)
            except StopIteration: return 0
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]; self.buf = self.buf[n:]
        return n

def _download_text_stream(conn_str, container, name):
    bsc = BlobServiceClient.from_connection_string(conn_str)
    bc = bsc.get_blob_client(container=container, blob=name)
    raw_iter = bc.download_blob(max_concurrency=1).chunks()
    return io.TextIOWrapper(io.BufferedReader(_IterStream(raw_iter), 8 * 1024 * 1024), encoding="utf-8")

def _iter_pairs(txt_stream):
    # top-level array: [ {"A":...,"B":...}, ... ]
    for pair in ijson.items(txt_stream, "item"):
        yield pair

# ---------- Append blob helpers ----------
def _get_append_client(conn_str: str, container: str, out_blob: str):
    bsc = BlobServiceClient.from_connection_string(conn_str)
    bc = bsc.get_blob_client(container=container, blob=out_blob)
    # Create append blob once (if not exists)
    try:
        bc.create_append_blob(content_settings=ContentSettings(content_type="application/json"))
    except ResourceExistsError:
        pass
    return bc

def _append_line(bc, line: str):
    # append_block max is 4MB — our lines are tiny
    data = (line if line.endswith("\n") else line + "\n").encode("utf-8")
    bc.append_block(data)

# ---------- Queue worker ----------
def main(msg: func.QueueMessage, queueOut: func.Out[str]):
    t0 = time.time()
    payload = json.loads(msg.get_body().decode())

    blob_full   = payload["blob"]                # "input-container/file.json"
    start_index = int(payload.get("start_index", 0))
    chunk_size  = int(payload.get("chunk_size", 200))
    out_blob    = payload["out_blob"]            # e.g., "result-file.jsonl"

    in_container, in_name = blob_full.split("/", 1)
    out_container = "output-container"           # change if you use a different container

    conn = os.environ["AzureWebJobsStorage"]
    logging.info("Chunk: %s from index=%d (chunk=%d) -> %s", blob_full, start_index, chunk_size, out_blob)

    # single output append blob
    out_client = _get_append_client(conn, out_container, out_blob)

    # stream input
    txt = _download_text_stream(conn, in_container, in_name)

    processed = 0
    idx = 0

    for pair in _iter_pairs(txt):
        if idx < start_index:
            idx += 1
            continue

        # Build matrices (float32 keeps memory lower)
        A_list, B_list = pair.get("A"), pair.get("B")
        ok = True
        err = None
        try:
            A = np.array(A_list, dtype=np.float32, copy=False)
            B = np.array(B_list, dtype=np.float32, copy=False)
            if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
                ok, err = False, f"invalid shapes {A.shape} vs {B.shape}"
        except Exception as e:
            ok, err = False, f"parse error: {e}"

        if ok:
            n = A.shape[0]
            C = _strassen_padded(A, B)  # <-- Strassen
            line = {
                "index": idx,
                "shape": [int(n), int(n)],
                "sum": float(np.sum(C)),
                "mean": float(np.mean(C)),
                "A_preview": (A[:JSONL_PREVIEW, :JSONL_PREVIEW].tolist() if n > JSONL_PREVIEW else A.tolist()),
                "B_preview": (B[:JSONL_PREVIEW, :JSONL_PREVIEW].tolist() if n > JSONL_PREVIEW else B.tolist()),
                "C_preview": (C[:JSONL_PREVIEW, :JSONL_PREVIEW].tolist() if n > JSONL_PREVIEW else C.tolist())
            }
        else:
            line = {"index": idx, "error": err}

        _append_line(out_client, json.dumps(line))

        # free quickly
        if ok:
            del A, B, C

        idx += 1
        processed += 1

        # Stop on budget → enqueue continuation
        if processed >= chunk_size or (time.time() - t0) > MAX_RUN_SECONDS:
            queueOut.set(json.dumps({
                "blob": blob_full,
                "start_index": idx,
                "chunk_size": chunk_size,
                "out_blob": out_blob
            }))
            logging.info("Re-queued continuation at index %d", idx)
            return

    # EOF — no requeue
    logging.info("Completed %s at last index %d → %s", blob_full, idx - 1, out_blob)
