import os
import json
import logging
import time   # 🔹 NEW: backoff rechecks
from typing import List

import azure.functions as func
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

TEMP_CONTAINER = "temp"
OUTPUT_CONTAINER = "output-container"
RUNS_PREFIX = "runs"                 # temp/runs/<run_id>/...
PARTS_DIR = "parts"                  # .../parts/
OUT_CT = "application/x-ndjson"

MAX_WAIT_SEC = int(os.getenv("MERGE_MAX_WAIT", "300"))  # 🔹 NEW: 5min wait cap
WAIT_INTERVAL = 10  # seconds between re-checks

def _bsc() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _list_files(bsc: BlobServiceClient, run_id: str, suffix: str) -> List[str]:
    cont = bsc.get_container_client(TEMP_CONTAINER)
    prefix = f"{RUNS_PREFIX}/{run_id}/{PARTS_DIR}/"
    names = []
    for b in cont.list_blobs(name_starts_with=prefix):
        if b.name.endswith(suffix):
            names.append(b.name)
    names.sort()
    return names

def _ensure_append_blob(bsc: BlobServiceClient, out_name: str):
    bc = bsc.get_blob_client(OUTPUT_CONTAINER, out_name)
    try:
        bc.create_append_blob(content_settings=ContentSettings(content_type=OUT_CT))
        logging.info("merge_result: created append blob %s/%s", OUTPUT_CONTAINER, out_name)
    except ResourceExistsError:
        pass
    return bc

def main(msg: func.QueueMessage) -> None:
    try:
        raw = msg.get_body()
        if isinstance(raw, (bytes, bytearray)):
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        else:
            payload = json.loads(raw)

        run_id = payload.get("run_id")
        expected = payload.get("expected_parts")  # splitter manifest gave this

        if not run_id:
            raise ValueError("run_id missing")

        bsc = _bsc()

        # 🔹 NEW: wait until every shard has a .done marker
        waited = 0
        while True:
            parts = _list_files(bsc, run_id, ".jsonl") + _list_files(bsc, run_id, ".ndjson")
            done  = _list_files(bsc, run_id, ".done")

            have = len(parts)
            finished = len(done)

            logging.info("merge_result: run=%s have_parts=%d done=%d expected=%s",
                         run_id, have, finished, expected)

            if expected is None:
                expected = finished  # fallback: trust done count

            if finished >= expected:
                break

            if waited >= MAX_WAIT_SEC:
                logging.warning("merge_result: timed out waiting for all done markers (waited %ds)", waited)
                return

            time.sleep(WAIT_INTERVAL)
            waited += WAIT_INTERVAL

        # 🔹 At this point, all expected .done files exist
        out_name = f"result-{run_id}.jsonl"
        bc_out = _ensure_append_blob(bsc, out_name)

        total_bytes = 0
        for name in parts:
            src = bsc.get_blob_client(TEMP_CONTAINER, name)
            stream = src.download_blob(max_concurrency=2)
            for chunk in stream.chunks():
                bc_out.append_block(chunk)
                total_bytes += len(chunk)

        logging.info("merge_result: merged %d parts (%d bytes) -> %s/%s",
                     len(parts), total_bytes, OUTPUT_CONTAINER, out_name)

    except Exception:
        logging.exception("merge_result: unhandled exception; msg body=%r", msg.get_body())
        raise
