import os
import io
import json
import time
import logging
from typing import List

import azure.functions as func
import numpy as np
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

# ---- Config ----
TEMP_CONTAINER    = os.getenv("TEMP_CONTAINER", "temp")
OUTPUT_CONTAINER  = os.getenv("OUTPUT_CONTAINER", "output-container")
RUNS_PREFIX       = "runs"
PARTS_DIR         = "parts"
RESULT_DIR        = "result"
OUTPUT_FORMAT     = os.getenv("OUTPUT_FORMAT", "npy").lower()  # "npy" (fast, default) or "npz" (smaller)

MAX_WAIT_SEC      = int(os.getenv("MERGE_MAX_WAIT", "300"))
WAIT_INTERVAL_SEC = int(os.getenv("MERGE_WAIT_INTERVAL", "10"))

OCTET_STREAM_CT   = "application/octet-stream"
PLAIN_CT          = "text/plain"
JSON_CT           = "application/json"

# ---- Helpers ----
def _bsc() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _list_suffix(bsc: BlobServiceClient, container: str, prefix: str, suffix: str) -> List[str]:
    names = []
    cont = bsc.get_container_client(container)
    for b in cont.list_blobs(name_starts_with=prefix):
        if b.name.endswith(suffix):
            names.append(b.name)
    names.sort()
    return names

def _read_meta(bsc: BlobServiceClient, run_id: str):
    meta_path = f"{RUNS_PREFIX}/{run_id}/meta.json"
    bc = bsc.get_blob_client(TEMP_CONTAINER, meta_path)
    data = bc.download_blob(max_concurrency=2).readall()
    meta = json.loads(data.decode("utf-8"))
    # Expect: shapeA [m,n], shapeB [n,p], num_shards, block, ...
    m, n = meta.get("shapeA", [None, None])
    _, p = meta.get("shapeB", [None, None])
    num_shards = int(meta.get("num_shards", 0))
    return int(m), int(p), num_shards, meta

def _upload_bytes(bsc: BlobServiceClient, container: str, name: str, data: bytes, content_type: str):
    bsc.get_blob_client(container, name).upload_blob(
        data, overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )

# ---- Main ----
def main(msg: func.QueueMessage) -> None:
    try:
        raw = msg.get_body()
        payload = json.loads(raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw)
        run_id = payload.get("run_id")
        if not run_id:
            raise ValueError("run_id missing in merge message")

        bsc = _bsc()

        # Read meta for shapes + expected shard count
        try:
            m, p, expected, meta = _read_meta(bsc, run_id)
            if not m or not p or not expected:
                raise ValueError("meta.json missing shapes or num_shards")
        except (ResourceNotFoundError, ValueError) as e:
            logging.error("merge_result: cannot read/parse meta.json for run=%s: %s", run_id, e)
            return  # nothing to do yet

        parts_prefix = f"{RUNS_PREFIX}/{run_id}/{PARTS_DIR}/"

        # Wait until all .done markers exist (one per original shard)
        waited = 0
        while True:
            done = _list_suffix(bsc, TEMP_CONTAINER, parts_prefix, ".done")
            finished = len(done)
            logging.info("merge_result: run=%s done=%d expected=%d", run_id, finished, expected)
            if finished >= expected:
                break
            if waited >= MAX_WAIT_SEC:
                logging.warning("merge_result: timed out waiting for done markers (waited %ds)", waited)
                return
            time.sleep(WAIT_INTERVAL_SEC)
            waited += WAIT_INTERVAL_SEC

        # Assemble final C from part-xxxxx.npz files
        C = np.zeros((m, p), dtype=np.float32)

        # We iterate deterministically over 1..expected (one part per original shard)
        for idx in range(1, expected + 1):
            part_name = f"{parts_prefix}part-{idx:05d}.npz"
            bc_part = bsc.get_blob_client(TEMP_CONTAINER, part_name)
            if not bc_part.exists():
                # If a shard produced no tiles (unlikely), just skip
                logging.info("merge_result: missing part npz %s (skipping)", part_name)
                continue

            data = bc_part.download_blob(max_concurrency=2).readall()
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                M = int(z["M"])
                i0 = z["i0"].astype(np.int64)
                j0 = z["j0"].astype(np.int64)
                bi = z["bi"].astype(np.int64)
                bj = z["bj"].astype(np.int64)
                for t in range(M):
                    C_tile = z[f"C_{t:03d}"]
                    ii0, jj0 = i0[t], j0[t]
                    bii, bjj = bi[t], bj[t]
                    C[ii0:ii0+bii, jj0:jj0+bjj] = C_tile  # last write wins

        # Write result (single blob) + success marker
        res_prefix = f"{RUNS_PREFIX}/{run_id}/{RESULT_DIR}"
        if OUTPUT_FORMAT == "npz":
            buf = io.BytesIO(); np.savez(buf, C=C); buf.seek(0)
            out_name = f"{res_prefix}/C.npz"
        else:
            buf = io.BytesIO(); np.save(buf, C); buf.seek(0)
            out_name = f"{res_prefix}/C.npy"

        _upload_bytes(bsc, TEMP_CONTAINER, out_name, buf.getvalue(), OCTET_STREAM_CT)
        _upload_bytes(bsc, TEMP_CONTAINER, f"{res_prefix}/_SUCCESS",
                      datetime.utcnow().isoformat().encode("utf-8"), PLAIN_CT)

        logging.info("merge_result: assembled C (%dx%d) -> %s/%s", m, p, TEMP_CONTAINER, out_name)

    except Exception:
        logging.exception("merge_result: unhandled exception; msg body=%r", msg.get_body())
        raise
