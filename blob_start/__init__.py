# blob_start/__init__.py
import os, datetime
from urllib.parse import quote
import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
import logging, uuid, time
def cd(**k):  # custom dimensions helper
    return {'custom_dimensions': k}

# helper: build a read SAS for videos-in/{name}
def sas_for_video(account_name: str, account_key: str, container: str, name: str, minutes: int = 120) -> str:
    expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=minutes)
    sas = generate_blob_sas(
        account_name=account_name,
        account_key=account_key,
        container_name=container,
        blob_name=name,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"https://{account_name}.blob.core.windows.net/{container}/{quote(name)}?{sas}"

# IMPORTANT: async + await
async def main(inputBlob: func.InputStream, starter: str) -> None:
    name = inputBlob.name.split("/", 1)[-1]  # "videos-in/<file>"
    name = name.split("/", 1)[-1]            # "<file>"

    run_id = str(uuid.uuid4())
    logging.info("blob_start.begin", extra=cd(runId=run_id, video=name))

    acct = os.environ["STORAGE_ACCOUNT_NAME"]
    key  = os.environ["STORAGE_ACCOUNT_KEY"]
    sas  = sas_for_video(acct, key, "videos-in", name)

    payload = {"runId": run_id,"name": name, "sas": sas}
    client = df.DurableOrchestrationClient(starter)
    instance_id = await client.start_new("orchestrator", None, payload)  # <-- await!

    logging.info(f"Started orchestration with ID = '{instance_id}' for blob '{name}'")
    logging.info("blob_start.end", extra=cd(runId=run_id, video=name, instanceId=instance_id))
