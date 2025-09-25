import json, os
from azure.storage.blob import BlobServiceClient
import logging, uuid, time
def cd(**k):  # custom dimensions helper
    return {'custom_dimensions': k}

def main(payload: dict):
    run_id = payload["runId"]
    name = payload["name"]
    events = payload["events"]
    logging.info("save_results.begin", extra=cd(runId=run_id, video=name, count=len(events)))

    conn = os.environ["AzureWebJobsStorage"]
    bsc = BlobServiceClient.from_connection_string(conn)
    blob = bsc.get_blob_client(container="insights", blob=f"{name}.json")
    doc = {"video": name, "count": len(events), "events": events}
    blob.upload_blob(json.dumps(doc, ensure_ascii=False, indent=2), overwrite=True)
    logging.info("save_results.end", extra=cd(runId=payload["runId"], video=payload["name"]))

