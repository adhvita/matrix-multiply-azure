# merge_result/__init__.py  (or wherever your merger lives)
import os, json, logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError

TEMP_CONTAINER   = "temp"
OUTPUT_CONTAINER = "output-container"

def _bsc() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _create_append(out_name: str):
    """
    Return a BlobClient for OUTPUT_CONTAINER/out_name.
    Ensures the target is an *append blob*. 
    """
    bc = _bsc().get_blob_client(OUTPUT_CONTAINER, out_name)
    try:
        bc.create_append_blob(
            content_settings=ContentSettings(content_type="application/x-ndjson")
        )
    except ResourceExistsError:
        pass  # already exists; fine
    return bc

def _list_parts(run_id: str):
    prefix = f"runs/{run_id}/parts/"
    cont = _bsc().get_container_client(TEMP_CONTAINER)
    names = [b.name for b in cont.list_blobs(name_starts_with=prefix)
             if b.name.endswith(".jsonl") or b.name.endswith(".ndjson") or b.name.endswith(".jsonl.gz")]
    names.sort()
    return names

def main(msg: func.QueueMessage, mergeOut: func.Out[str]):
    payload  = json.loads(msg.get_body().decode("utf-8"))
    run_id   = payload["run_id"]
    expected = payload.get("expected_parts")

    parts = _list_parts(run_id)
    have  = len(parts)
    logging.info("merge_result: run=%s have=%d expected=%s", run_id, have, expected)

    if expected is None or have < expected:
        # Re-enqueue ourselves to poll again later
        mergeOut.set(json.dumps({"run_id": run_id, "expected_parts": expected}))
        return

    out_name = f"result-{run_id}.jsonl"
    out = _create_append(out_name)

    bsc = _bsc()
    total_bytes = 0
    for name in parts:
        src = bsc.get_blob_client(TEMP_CONTAINER, name)
        # stream chunks to keep memory low
        for chunk in src.download_blob(max_concurrency=2).chunks():
            out.append_block(chunk)
            total_bytes += len(chunk)

    logging.info("merge_result: merged %d parts (%d bytes) -> %s/%s",
                 len(parts), total_bytes, OUTPUT_CONTAINER, out_name)
