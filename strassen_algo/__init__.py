import os
import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, Tuple

import azure.functions as func
from azure.storage.blob import BlobServiceClient
import ijson
import numpy as np

from .strassen_module import strassen

# ---- tunables -------------------------------------------------
MAX_BYTES_FOR_INMEM = 200 * 1024 * 1024  # if result C bigger than this, summarize
PREVIEW_SIZE = 8
BLOCK = 512
# ---------------------------------------------------------------

def blocked_matmul(A: np.ndarray, B: np.ndarray, block: int = BLOCK) -> np.ndarray:
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(0, n, block):
        i2 = min(i + block, n)
        for j in range(0, n, block):
            j2 = min(j + block, n)
            C[i:i2, j:j2] = 0
            for k in range(0, n, block):
                k2 = min(k + block, n)
                C[i:i2, j:j2] += A[i:i2, k:k2] @ B[k:k2, j:j2]
    return C

def safe_multiply(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, bool]:
    n = A.shape[0]
    bytes_c = n * n * A.dtype.itemsize
    if bytes_c <= MAX_BYTES_FOR_INMEM:
        try:
            C = strassen(A, B) if n >= 256 else (A @ B)
        except Exception:
            C = A @ B
        return C, True
    else:
        return blocked_matmul(A, B, block=BLOCK), False

def summarize_matrix(C: np.ndarray, preview: int = PREVIEW_SIZE) -> Dict[str, Any]:
    n = C.shape[0]; p = min(preview, n)
    return {
        "shape": [int(n), int(n)],
        "dtype": str(C.dtype),
        "preview_top_left": C[:p, :p].tolist(),
        "sum": float(np.sum(C)),
        "mean": float(np.mean(C)),
    }

# ---- helper: turn chunk iterator into a readable stream --------
class IterStream(io.RawIOBase):
    def __init__(self, iterator):
        self._it = iter(iterator)
        self._buf = b""
    def readable(self): return True
    def readinto(self, b):
        if not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n
# ---------------------------------------------------------------

def iter_pairs(buffered: io.BufferedReader) -> Iterable[Dict[str, Any]]:
    # JSON file is expected to be a top-level array: [ { "A":..., "B":... }, ... ]
    for pair in ijson.items(buffered, "item"):
        yield pair

def main(triggerblob: func.InputStream, outputblob: func.Out[str]) -> None:
    try:
        # Only use triggerblob for the name; do NOT read it (host may buffer).
        blob_name = triggerblob.name.split("/", 1)[-1]  # "<path in container>"
        logging.info("Triggered on: %s (length=%s)", triggerblob.name, getattr(triggerblob, "length", "n/a"))

        # Stream from Storage directly (no buffering)
        conn_str = os.environ["AzureWebJobsStorage"]
        input_container = "input-container"   # <-- match function.json path
        bsc = BlobServiceClient.from_connection_string(conn_str)
        bc = bsc.get_blob_client(container=input_container, blob=blob_name)

        # chunks() yields small byte chunks; wrap into a file-like stream
        raw = IterStream(bc.download_blob().chunks())
        buffered = io.BufferedReader(raw)

        results = []
        count = 0

        for idx, pair in enumerate(iter_pairs(buffered)):
            if "A" not in pair or "B" not in pair:
                results.append({"index": idx, "error": "Missing keys A or B"})
                continue

            A = np.array(pair["A"], dtype=np.float32)
            B = np.array(pair["B"], dtype=np.float32)

            if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
                results.append({"index": idx, "error": f"Invalid shapes: {A.shape} vs {B.shape}"})
                continue

            n = A.shape[0]
            logging.info("Processing pair %d, shape %s", idx, A.shape)

            C, return_full = safe_multiply(A, B)

            if return_full and n <= PREVIEW_SIZE:
                results.append({
                    "index": idx,
                    "shape": [int(n), int(n)],
                    "A": A.tolist(),
                    "B": B.tolist(),
                    "C": C.tolist()
                })
            else:
                results.append({
                    "index": idx,
                    "shape": [int(n), int(n)],
                    "A_preview": (A[:PREVIEW_SIZE, :PREVIEW_SIZE].tolist() if n > PREVIEW_SIZE else A.tolist()),
                    "B_preview": (B[:PREVIEW_SIZE, :PREVIEW_SIZE].tolist() if n > PREVIEW_SIZE else B.tolist()),
                    "C_summary": summarize_matrix(C, PREVIEW_SIZE)
                })

            del A, B, C
            count += 1

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_pairs": count,
            "results": results
        }
        outputblob.set(json.dumps(payload))
        logging.info("✅ Completed %d pairs for %s", count, blob_name)

    except Exception:
        logging.exception("❌ Error during streaming processing")
        raise


# def main(inputblob: func.InputStream, outputblob: func.Out[str]):
#     try:
#         time.sleep(10)
#         logging.info("Function triggered: Reading input blob '%s'", inputblob.name)
#         # Step 1: Read the full synthetic_dataset.json content
#         logging.info("Function triggered")
#         content = inputblob.read().decode('utf-8')
#         matrix_pairs = json.loads(content)

#         results = []

#         # Step 2: Loop through each A-B pair
#         for idx, pair in enumerate(matrix_pairs):
#             A = np.array(pair["A"])
#             B = np.array(pair["B"])

#             if A.shape != B.shape or A.shape[0] != A.shape[1]:
#                 logging.info("Processing matrix pair %d: shape %s", idx, A.shape)
#                 results.append({
#                     "index": idx,
#                     "error": "Matrix shapes are invalid or mismatched."
#                 })
#                 continue
            
#             logging.info("Processing matrix pair %d: shape %s", idx, A.shape)
#             C = strassen(A, B)
#             results.append({
#                 "index": idx,
#                 "A": pair["A"],
#                 "B": pair["B"],
#                 "C": C.tolist()
#             })

#         # Step 3: Save results to output blob
#         output_data = {
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#             "num_pairs": len(results),
#             "results": results
#         }
#         logging.info("✅ Processed %d matrix pairs from blob '%s'", len(results), inputblob.name)

#         outputblob.set(json.dumps(output_data, indent=2))
#         logging.info("Processed %d matrix pairs from %s",
#                      len(results), inputblob.name)
    
#     except Exception as e:
#         logging.exception(" Error processing synthetic dataset")
#         raise
