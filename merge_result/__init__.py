import os
import json
import logging
from typing import List

import azure.functions as func
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

TEMP_CONTAINER = "temp"
OUTPUT_CONTAINER = "output-container"
RUNS_PREFIX = "runs"                 # temp/runs/<run_id>/...
PARTS_DIR = "parts"                  # .../parts/
OUT_CT = "application/x-ndjson"      # or "application/jsonl"

def _bsc() -> BlobServiceClient:
    # Will raise KeyError if missing; we catch it and log clearly.
    conn = os.environ["AzureWebJobsStorage"]
    return BlobServiceClient.from_connection_string(conn)

def _list_parts(bsc: BlobServiceClient, run_id: str) -> List[str]:
    """
    Returns sorted list of part blob names:
    temp/runs/<run_id>/parts/part-00001.jsonl, ...
    """
    cont = bsc.get_container_client(TEMP_CONTAINER)
    prefix = f"{RUNS_PREFIX}/{run_id}/{PARTS_DIR}/"
    names = []
    # name_starts_with avoids scanning whole container
    for b in cont.list_blobs(name_starts_with=prefix):
        if b.name.endswith(".jsonl") or b.name.endswith(".ndjson"):
            names.append(b.name)
    names.sort()
    return names

def _ensure_append_blob(bsc: BlobServiceClient, out_name: str):
    """
    Creates (or reuses) an append blob in OUTPUT_CONTAINER and
    returns its BlobClient.
    """
    bc = bsc.get_blob_client(OUTPUT_CONTAINER, out_name)
    try:
        bc.create_append_blob(
            content_settings=ContentSettings(content_type=OUT_CT)
        )
        logging.info("merge_result: created append blob %s/%s", OUTPUT_CONTAINER, out_name)
    except ResourceExistsError:
        # OK to reuse – append blob already exists
        pass
    return bc

def main(msg: func.QueueMessage) -> None:
    try:
        raw = msg.get_body()
        # Defensive: msg.get_body() may be bytes
        try:
            if isinstance(raw, (bytes, bytearray)):
                payload = json.loads(raw.decode("utf-8", errors="replace"))
            else:
                payload = json.loads(raw)
        except Exception as e:
            logging.exception("merge_result: failed to parse queue message body: %r", raw)
            # Do not rethrow raw; raise to push to poison (it’s a bad message)
            raise

        run_id = payload.get("run_id")
        expected = payload.get("expected_parts")  # can be None

        if not run_id or not isinstance(run_id, str):
            logging.error("merge_result: invalid or missing run_id in payload: %r", payload)
            raise ValueError("run_id missing/invalid")

        bsc = _bsc()

        parts = _list_parts(bsc, run_id)
        have = len(parts)
        logging.info("merge_result: run_id=%s have=%d expected=%s parts=%r",
                     run_id, have, expected, parts)

        # If producer didn’t compute expected_parts, we can merge when we see >=1
        if expected is None:
            expected = 1

        if have < expected:
            # Not ready yet — re-enqueue the *same* message to merge later.
            # Use the same queue by writing back (SDK) or leverage retry strategy on producer.
            # Minimal: just log and return; whoever enqueued this will requeue again.
            logging.info("merge_result: not enough parts yet (have=%d < expected=%d); will be re-queued by producer", have, expected)
            return

        # Ready: merge all parts
        out_name = f"result-{run_id}.jsonl"
        bc_out = _ensure_append_blob(bsc, out_name)

        total_bytes = 0
        for name in parts:
            src = bsc.get_blob_client(TEMP_CONTAINER, name)
            # stream in chunks; small file so max_concurrency=2 is fine
            stream = src.download_blob(max_concurrency=2)
            for chunk in stream.chunks():
                bc_out.append_block(chunk)
                total_bytes += len(chunk)

        logging.info("merge_result: merged %d parts (%d bytes) -> %s/%s",
                     len(parts), total_bytes, OUTPUT_CONTAINER, out_name)

    except Exception:
        # CRITICAL: capture full stack. This is what you were missing; without it
        # Azure shows only “Function failed” with no Python traceback.
        logging.exception("merge_result: unhandled exception; message body=%r", msg.get_body())
        # Re-raise so Functions runtime marks the dequeue as a failure.
        # After MaxDequeueCount it will go to poison queue (by design).
        raise
