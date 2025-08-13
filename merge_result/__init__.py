import os
import json
import logging
from typing import List, Tuple

import azure.functions as func
from azure.storage.blob import (
    BlobServiceClient, ContainerClient, AppendBlobClient, BlobClient, ContentSettings
)

# Config via App Settings when possible; defaults match what you’ve been using
TEMP_CONTAINER   = os.getenv("TEMP_CONTAINER",   "temp")
OUTPUT_CONTAINER = os.getenv("OUT_CONTAINER",    "output-container")
PARTS_DIR_ENV    = os.getenv("PARTS_DIR",        "parts")  # we’ll fall back to "shards" if empty


def _svc() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])


def _ensure_container(svc: BlobServiceClient, name: str) -> ContainerClient:
    cc = svc.get_container_client(name)
    try:
        cc.create_container()
        logging.info("merge_result: created container %s", name)
    except Exception:
        pass
    return cc


def _ensure_append_blob(cc: ContainerClient, name: str) -> AppendBlobClient:
    abc = cc.get_append_blob_client(name)
    if not abc.exists():
        # content type is nice-to-have; append blob supports it
        abc.create_blob(content_settings=ContentSettings(content_type="application/json"))
        logging.info("merge_result: created append blob %s/%s", cc.container_name, name)
    return abc


def _list_pairs(container: ContainerClient, run_id: str, base_dir: str) -> List[str]:
    """
    Return sorted list of part blob names for runs/<run_id>/<base_dir>/.
    Accept both .ndjson and .jsonl.
    """
    prefix = f"runs/{run_id}/{base_dir}/"
    names: List[str] = []
    for b in container.list_blobs(name_starts_with=prefix):
        if b.name.endswith(".ndjson") or b.name.endswith(".jsonl"):
            names.append(b.name)

    # Prefer numeric order if names look like part-00001.jsonl, otherwise plain sort
    def _key(n: str) -> Tuple[int, str]:
        # extract digits after "part-" if present
        try:
            base = os.path.basename(n)
            stem = os.path.splitext(base)[0]  # "part-00001"
            if stem.startswith("part-"):
                num = int(stem.split("part-")[1])
                return (num, n)
        except Exception:
            pass
        return (10**9, n)  # non-part names go to the end but remain deterministic

    names.sort(key=_key)
    return names


def main(msg: func.QueueMessage, mergeOut: func.Out[str]) -> None:
    payload = json.loads(msg.get_body().decode("utf-8"))
    run_id   = payload["run_id"]
    expected = payload.get("expected_parts")

    logging.info(
        "merge_result: start run_id=%s TEMP=%s OUT=%s PARTS_DIR=%s expected=%s",
        run_id, TEMP_CONTAINER, OUTPUT_CONTAINER, PARTS_DIR_ENV, expected
    )

    svc      = _svc()
    temp_cc  = _ensure_container(svc, TEMP_CONTAINER)
    out_cc   = _ensure_container(svc, OUTPUT_CONTAINER)

    # 1) find parts in configured dir; if none, fall back to "shards"
    parts = _list_pairs(temp_cc, run_id, PARTS_DIR_ENV)
    if not parts and PARTS_DIR_ENV != "shards":
        alt = "shards"
        parts = _list_pairs(temp_cc, run_id, alt)
        if parts:
            logging.warning("merge_result: no files in '%s', using fallback '%s'", PARTS_DIR_ENV, alt)

    logging.info("merge_result: discovered %d part(s)", len(parts))
    for p in parts:
        logging.info("merge_result: part -> %s", p)

    have = len(parts)

    # 2) only requeue if caller specified an expected count and we’re under it
    if expected is not None and have < expected:
        logging.info(
            "merge_result: have %d < expected %d -> requeue",
            have, expected
        )
        mergeOut.set(json.dumps({"run_id": run_id, "expected_parts": expected}))
        return

    # If expected is None: merge NOW with everything we found (your previous code re-queued forever)
    if have == 0:
        # Nothing to merge yet -> let queue retry later instead of writing an empty output
        raise RuntimeError(f"merge_result: no parts available for run {run_id}; will retry")

    # 3) prepare target append blob
    out_name = f"result-{run_id}.jsonl"
    abc = _ensure_append_blob(out_cc, out_name)

    # Log size before append
    pre_size = (abc.get_blob_properties().size or 0)
    logging.info("merge_result: target %s/%s pre-size=%d", OUTPUT_CONTAINER, out_name, pre_size)

    # 4) append each part streaming in chunks
    appended_bytes = 0
    for name in parts:
        bc: BlobClient = temp_cc.get_blob_client(name)
        stream = bc.download_blob(max_concurrency=2)

        # append in chunks (the iterator yields bytes)
        for chunk in stream.chunks():
            if not chunk:
                continue
            abc.append_block(chunk)
            appended_bytes += len(chunk)

        logging.info("merge_result: appended part %s", name)

    post_size = (abc.get_blob_properties().size or 0)
    logging.info(
        "merge_result: merged %d part(s) -> %s/%s; bytes_appended=%d final_size=%d delta=%d",
        have, OUTPUT_CONTAINER, out_name, appended_bytes, post_size, post_size - pre_size
    )
