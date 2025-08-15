import os
import json
import logging
from datetime import datetime, timezone

import azure.functions as func
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings
from ..strassen_algo.strassen_module import strassen

TEMP_CONTAINER = "temp"

def _cs() -> str:
    cs = os.environ.get("AzureWebJobsStorage")
    if not cs:
        raise RuntimeError("AzureWebJobsStorage not set")
    return cs

def main(msg: func.QueueMessage, mergeOut: func.Out[str]) -> None:
    try:
        payload = json.loads(msg.get_body().decode("utf-8"))
        run_id     = payload["run_id"]
        shard_no   = int(payload["shard_no"])
        shard_blob = payload["shard_blob"]  # "temp/runs/<run_id>/shards/shard-00001.ndjson"

        logging.info("process_chunk start run=%s shard=%d shard_blob=%s", run_id, shard_no, shard_blob)

        container, name = shard_blob.split("/", 1) if "/" in shard_blob else (TEMP_CONTAINER, shard_blob)

        # Build a fresh BlobServiceClient right here (no helpers, no reuse).
        bsc = BlobServiceClient.from_connection_string(_cs())
        logging.info("bsc type=%s", type(bsc))  # should be BlobServiceClient

        # Input stream
        bc_in = bsc.get_blob_client(container=container, blob=name)
        downloader = bc_in.download_blob(max_concurrency=2)
        stream = downloader.chunks()

        # Output append blob temp/runs/<run_id>/parts/part-xxxxx.jsonl
        part_name = f"runs/{run_id}/parts/part-{shard_no:05d}.jsonl"
        bc_out = bsc.get_blob_client(container=TEMP_CONTAINER, blob=part_name)
        try:
            bc_out.create_append_blob(content_settings=ContentSettings(content_type="application/x-ndjson"))
        except Exception:
            pass  # already exists

        buf = bytearray()
        count = 0

        def flush_lines():
            nonlocal buf, count
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = buf[:nl].decode("utf-8").strip()
                del buf[:nl+1]
                if not line:
                    continue

                try:
                    pair = json.loads(line)
                    A = np.array(pair["A"], dtype=np.float32)
                    B = np.array(pair["B"], dtype=np.float32)
                    C = strassen(A, B)
                    row = {
                        "index": count,
                        "shape": [int(A.shape[0]), int(A.shape[1])],
                        "sum": float(np.sum(C)),
                        "mean": float(np.mean(C)),
                    }
                except Exception as ex:
                    row = {"index": count, "error": str(ex)}

                bc_out.append_block((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
                count += 1

        for chunk in stream:
            buf.extend(chunk)
            flush_lines()

        flush_lines()

        logging.info("process_chunk done run=%s shard=%d wrote %s (records=%d)",
                     run_id, shard_no, part_name, count)

        mergeOut.set(json.dumps({
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat()
        }))

    except Exception:
        logging.exception("process_chunk failed")
        raise
