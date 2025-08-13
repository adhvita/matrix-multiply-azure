import os, json, logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings

TEMP_CONTAINER   = "temp"
OUTPUT_CONTAINER = "output-container"

def _bsc():
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _list_parts(run_id):
    prefix = f"runs/{run_id}/parts/"
    cont = _bsc().get_container_client(TEMP_CONTAINER)
    names = [b.name for b in cont.list_blobs(name_starts_with=prefix) if b.name.endswith(".jsonl")]
    names.sort()  # part-00001.jsonl, part-00002.jsonl, ...
    return names

def _create_append(out_name):
    bc = _bsc().get_blob_client(OUTPUT_CONTAINER, out_name)
    try:
        bc.create_append_blob(content_settings=ContentSettings(content_type="application/json"))
    except Exception:
        pass
    return bc  

def main(msg: func.QueueMessage, mergeOut: func.Out[str]):
    payload = json.loads(msg.get_body().decode())
    run_id = payload["run_id"]
    expected = payload.get("expected_parts")

    parts = _list_parts(run_id)
    have = len(parts)
    logging.info("merge check run=%s have=%d expected=%s", run_id, have, expected)

    if expected is None or have < expected:
        # poll again shortly
        mergeOut.set(json.dumps({ "run_id": run_id, "expected_parts": expected }))
        return

    out_name = f"result-{run_id}.jsonl"
    bc_out = _create_append(out_name)
    bsc = _bsc()
    for name in parts:
        src = bsc.get_blob_client(TEMP_CONTAINER, name)
        for chunk in src.download_blob(max_concurrency=2).chunks():
            bc_out.append_block(chunk)

    logging.info("Merged %d parts -> %s/%s", len(parts), OUTPUT_CONTAINER, out_name)
