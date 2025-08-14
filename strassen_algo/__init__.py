# strassen_algo/__init__.py  (splitter)
import io
import json
import os
import uuid
import logging
from datetime import datetime, timezone

import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
import ijson

# -----------------------
# Tunables / Constants
# -----------------------
LINES_PER_SHARD = 50          # how many items per NDJSON shard
READ_BUF_MB     = 16               # streaming read buffer (MiB)
TEMP_CONTAINER  = "temp"           # temp container that holds runs/<run_id>/

# -----------------------
# Helpers
# -----------------------
def _bsc() -> BlobServiceClient:
    """BlobServiceClient from the Functions storage connection."""
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _iter_text(bc) -> io.TextIOWrapper:
    """
    Given a BlobClient `bc`, return a TextIO stream (utf-8) that yields data by
    streaming chunks from the blob (low memory).
    """
    chunks = bc.download_blob(max_concurrency=2).chunks()

    class _Raw(io.RawIOBase):
        def __init__(self, it):
            self.it = iter(it)
            self.buf = b""

        def readable(self) -> bool:  # type: ignore[override]
            return True

        def readinto(self, b) -> int:  # type: ignore[override]
            if not self.buf:
                try:
                    self.buf = next(self.it)
                except StopIteration:
                    return 0
            n = min(len(b), len(self.buf))
            b[:n] = self.buf[:n]
            self.buf = self.buf[n:]
            return n

    return io.TextIOWrapper(
        io.BufferedReader(_Raw(chunks), READ_BUF_MB * 1024 * 1024),
        encoding="utf-8",
    )

# -----------------------
# Entry point
# -----------------------
def main(inBlob: func.InputStream, queueOut: func.Out[str]) -> None:
    """
    Blob-triggered splitter.

    - Reads a large JSON array of matrix-pairs from the input blob (streaming).
    - Writes NDJSON shard files to temp/runs/<run_id>/shards/shard-00001.ndjson ...
    - Enqueues one message per shard to the `matrix-shards` queue via queueOut.
    - Writes temp/runs/<run_id>/manifest.json with `expected_parts` for the merger.
    """
    # Derive a unique run id from the file name + nonce
    base = os.path.splitext(os.path.basename(inBlob.name))[0]
    run_id = f"{base}-{uuid.uuid4().hex[:8]}"

    # Stream input JSON from the original blob
    in_container, in_name = inBlob.name.split("/", 1)
    bc_in = _bsc().get_blob_client(in_container, in_name)
    txt = _iter_text(bc_in)  # text stream (utf-8), safe for ijson

    # Where shards will be written
    shard_prefix = f"runs/{run_id}/shards"
    shard_idx = 0
    total_items = 0
    lines: list[str] = []

    def flush_shard():
        nonlocal shard_idx, lines
        if not lines:
            return
        shard_idx += 1
        shard_path = f"{shard_prefix}/shard-{shard_idx:05d}.ndjson"

        # Write NDJSON shard
        data = ("\n".join(lines) + "\n").encode("utf-8")
        _bsc().get_blob_client(TEMP_CONTAINER, shard_path).upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/x-ndjson"),
        )

        # Enqueue message for worker (process_chunk)
        queueOut.set(json.dumps({
            "run_id": run_id,
            "shard_no": shard_idx,
            # include container so the worker can open directly:
            "shard_blob": f"{TEMP_CONTAINER}/{shard_path}",
        }))

        lines.clear()

    # Stream the top-level JSON array: [ { "A":..., "B":... }, ... ]
    for item in ijson.items(txt, "item"):
        # Re-emit as compact JSON line (the worker can parse it)
        lines.append(json.dumps(item, separators=(",", ":")))
        total_items += 1
        if len(lines) >= LINES_PER_SHARD:
            flush_shard()

    # Flush tail
    flush_shard()

    # Write a small manifest for the run (useful for the merger)
    manifest = {
        "run_id": run_id,
        "expected_parts": shard_idx,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _bsc().get_blob_client(TEMP_CONTAINER, f"runs/{run_id}/manifest.json").upload_blob(
        json.dumps(manifest).encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

    logging.info(
        "Split complete: run=%s shards=%d total_items=%d",
        run_id, shard_idx, total_items
    )
