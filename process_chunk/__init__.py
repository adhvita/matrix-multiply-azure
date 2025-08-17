import os
import json
import logging
import time   # 🔹 NEW: for time budget
from datetime import datetime, timezone

import azure.functions as func
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient   # 🔹 NEW: for continuation shards
from ..strassen_algo.strassen_module import strassen

TEMP_CONTAINER = "temp"
PROCESS_QUEUE  = os.getenv("PROCESS_QUEUE", "process-queue")   # 🔹 NEW

# 🔹 NEW: time-budget constants
MAX_SECONDS = int(os.getenv("CHUNK_TIME_BUDGET_SEC", "240"))  # 4 minutes
SAFETY_MARGIN = int(os.getenv("CHUNK_SAFETY_MARGIN_SEC", "15"))

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
        shard_blob = payload["shard_blob"]

        logging.info("process_chunk start run=%s shard=%d shard_blob=%s", run_id, shard_no, shard_blob)

        container, name = shard_blob.split("/", 1) if "/" in shard_blob else (TEMP_CONTAINER, shard_blob)

        bsc = BlobServiceClient.from_connection_string(_cs())
        bc_in = bsc.get_blob_client(container=container, blob=name)
        stream = bc_in.download_blob(max_concurrency=2).chunks()

        # 🔹 Output append blob (one per original shard)
        part_name = f"runs/{run_id}/parts/part-{shard_no:05d}.jsonl"
        bc_out = bsc.get_blob_client(container=TEMP_CONTAINER, blob=part_name)
        try:
            bc_out.create_append_blob(content_settings=ContentSettings(content_type="application/x-ndjson"))
        except Exception:
            pass

        started = time.monotonic()
        buf = bytearray()
        tiles = {}           # 🔹 NEW: (i0,j0) → accumulated C_tile
        leftover_lines = []  # 🔹 NEW: lines we couldn’t process before timeout

        def time_left():
            return MAX_SECONDS - (time.monotonic() - started)

        def handle_line(line: str):
            obj = json.loads(line)
            # 🔹 Detect tile-mode vs pair-mode
            if "i0" in obj and "j0" in obj:
                i0, j0 = int(obj["i0"]), int(obj["j0"])
                A = np.array(obj["A"], dtype=np.float32)
                B = np.array(obj["B"], dtype=np.float32)
                C_part = strassen(A, B)
                key = (i0, j0)
                tiles[key] = C_part if key not in tiles else tiles[key] + C_part
            else:
                # array-of-pairs fallback
                A = np.array(obj["A"], dtype=np.float32)
                B = np.array(obj["B"], dtype=np.float32)
                C = strassen(A, B)
                row = {
                    "index": obj.get("index", 0),
                    "shape": [int(A.shape[0]), int(A.shape[1])],
                    "sum": float(np.sum(C)),
                    "mean": float(np.mean(C)),
                }
                bc_out.append_block((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))

        # 🔹 Stream and stop early if close to timeout
        for chunk in stream:
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = buf[:nl].decode("utf-8").strip()
                del buf[:nl+1]
                if not line:
                    continue
                if time_left() < SAFETY_MARGIN:
                    leftover_lines.append(line)
                    # save rest of buf as leftover
                    if buf:
                        leftover_lines.append(buf.decode("utf-8"))
                        buf.clear()
                    break
                handle_line(line)
            if leftover_lines:
                break

        # 🔹 flush any leftover line after stream ends
        if buf and not leftover_lines:
            line = buf.decode("utf-8").strip()
            if line:
                handle_line(line)

        # 🔹 Write tile-mode results (one row per (i,j))
        for (i0, j0), C in tiles.items():
            row = {
                "run_id": run_id,
                "i0": i0,
                "j0": j0,
                "bi": int(C.shape[0]),
                "bj": int(C.shape[1]),
                "C": C.tolist()
            }
            bc_out.append_block((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))

        # 🔹 If leftover lines, enqueue a continuation shard
        if leftover_lines:
            cont_blob = f"runs/{run_id}/shards/shard-{shard_no:05d}-cont-{int(time.time())}.ndjson"
            bsc.get_blob_client(TEMP_CONTAINER, cont_blob).upload_blob(
                ("\n".join([l for l in leftover_lines if l]) + "\n").encode("utf-8"),
                overwrite=True,
                content_settings=ContentSettings(content_type="application/x-ndjson")
            )
            qc = QueueClient.from_connection_string(_cs(), PROCESS_QUEUE)
            qc.create_queue()
            qc.send_message(json.dumps({
                "run_id": run_id,
                "shard_no": shard_no,
                "shard_blob": f"{TEMP_CONTAINER}/{cont_blob}"
            }))
            logging.info("Continuation shard queued: %s", cont_blob)
        else:
            # 🔹 Mark this shard as done
            done_name = f"runs/{run_id}/parts/part-{shard_no:05d}.done"
            bsc.get_blob_client(TEMP_CONTAINER, done_name).upload_blob(
                b"ok", overwrite=True,
                content_settings=ContentSettings(content_type="text/plain")
            )

        logging.info("process_chunk done run=%s shard=%d wrote %s",
                     run_id, shard_no, part_name)

        mergeOut.set(json.dumps({
            "run_id": run_id,
            "shard_no": shard_no,
            "part_blob": f"{TEMP_CONTAINER}/{part_name}",
            "ts": datetime.now(timezone.utc).isoformat()
        }))

    except Exception:
        logging.exception("process_chunk failed")
        raise
