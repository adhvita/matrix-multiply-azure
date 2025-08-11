import logging
import json
import numpy as np
import time
import azure.functions as func
from .strassen_module import strassen
from datetime import datetime, timezone

import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, Tuple

import azure.functions as func
import ijson
import numpy as np

# Your Strassen implementation
from .strassen_module import strassen

# --- Tunables ---------------------------------------------------------------

# When n*n*elem_size exceeds this many bytes, avoid allocating full C in RAM.
MAX_BYTES_FOR_INMEM = 200 * 1024 * 1024  # 200 MB

# For big results, include only a small preview and checksums
PREVIEW_SIZE = 8  # 8x8 preview

# Block size for blocked multiply (keeps peak RAM low)
BLOCK = 512

# ---------------------------------------------------------------------------

def blocked_matmul(A: np.ndarray, B: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Memory-friendlier multiply than one big matmul; still returns full C."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(0, n, block):
        i2 = min(i + block, n)
        for j in range(0, n, block):
            j2 = min(j + block, n)
            # zero tile
            C[i:i2, j:j2] = 0
            for k in range(0, n, block):
                k2 = min(k + block, n)
                C[i:i2, j:j2] += A[i:i2, k:k2] @ B[k:k2, j:j2]
    return C


def safe_multiply(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Multiply choosing algorithm based on size.
    Returns (C, full_return) where full_return indicates whether we should serialize the full C.
    """
    assert A.shape == B.shape and A.shape[0] == A.shape[1]
    n = A.shape[0]
    elem_size = A.dtype.itemsize
    bytes_c = n * n * elem_size

    # Use Strassen for smaller sizes, else blocked matmul.
    if bytes_c <= MAX_BYTES_FOR_INMEM:
        # Heuristic: Strassen typically helps for n >= 256–512; your impl decides.
        try:
            if n >= 256:
                C = strassen(A, B)
            else:
                C = A @ B
        except Exception:
            # Fallback if Strassen fails for any reason
            C = A @ B
        return C, True
    else:
        # Too big to safely keep full C in memory; still compute but only return summaries.
        C = blocked_matmul(A, B, block=BLOCK)
        return C, False


def summarize_matrix(C: np.ndarray, preview: int = PREVIEW_SIZE) -> Dict[str, Any]:
    """Return small preview and checksums to keep output light."""
    n = C.shape[0]
    p = min(preview, n)
    preview_block = C[:p, :p].tolist()
    return {
        "shape": [int(n), int(n)],
        "dtype": str(C.dtype),
        "preview_top_left": preview_block,
        "sum": float(np.sum(C)),
        "mean": float(np.mean(C)),
    }


def iter_pairs(buf: io.BufferedReader) -> Iterable[Dict[str, Any]]:
    """
    Stream the top-level JSON array:  [ {"A": ... , "B": ...}, {"A": ... , "B": ...}, ... ]
    Each yielded item is one pair dict (ijson builds only that item into memory).
    """
    for pair in ijson.items(buf, "item"):
        yield pair


def main(inputblob: func.InputStream, outputblob: func.Out[str]) -> None:
    try:
        logging.info("Function triggered on: %s  size=%d bytes", inputblob.name, inputblob.length)

        # IMPORTANT: wrap the blob stream so ijson can incrementally parse it
        buffered = io.BufferedReader(inputblob)

        results = []
        count = 0

        for idx, pair in enumerate(iter_pairs(buffered)):
            # Expecting each pair to have lists for A and B
            if "A" not in pair or "B" not in pair:
                results.append({"index": idx, "error": "Missing keys A or B"})
                continue

            # Convert to float32 to halve memory vs float64 (tweak if you need integers)
            A = np.array(pair["A"], dtype=np.float32)
            B = np.array(pair["B"], dtype=np.float32)

            if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
                results.append({"index": idx, "error": f"Invalid shapes: {A.shape} vs {B.shape}"})
                continue

            n = A.shape[0]
            logging.info("Processing pair %d with shape %s", idx, A.shape)

            C, can_return_full = safe_multiply(A, B)

            if can_return_full and n <= PREVIEW_SIZE:  # tiny matrices → include full C
                results.append({
                    "index": idx,
                    "shape": [int(n), int(n)],
                    "A": pair["A"],
                    "B": pair["B"],
                    "C": C.astype(np.float32).tolist(),
                })
            else:
                # keep response small
                results.append({
                    "index": idx,
                    "shape": [int(n), int(n)],
                    "A_preview": (A[:PREVIEW_SIZE, :PREVIEW_SIZE].tolist() if n > PREVIEW_SIZE else A.tolist()),
                    "B_preview": (B[:PREVIEW_SIZE, :PREVIEW_SIZE].tolist() if n > PREVIEW_SIZE else B.tolist()),
                    "C_summary": summarize_matrix(C, PREVIEW_SIZE),
                })

            # free per-iteration arrays promptly
            del A, B, C
            count += 1

        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_pairs": count,
            "results": results,
        }

        outputblob.set(json.dumps(output_data))
        logging.info("✅ Completed %d pairs from %s", count, inputblob.name)

    except Exception as e:
        logging.exception("❌ Error processing streamed dataset")
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
