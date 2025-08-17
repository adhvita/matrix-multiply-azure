import os, io, json, uuid, logging
import azure.functions as func
import ijson
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient, ContentSettings

# ------- Tunables (override at runtime via env if you want) -------
# Hard cap on rows per shard (fallback if matrices are tiny)
DEFAULT_MAX_LINES   = int(os.getenv("SHARD_MAX_LINES", "400"))
# Safety cap on shard payload size (bytes) to avoid big uploads
DEFAULT_MAX_BYTES   = int(os.getenv("SHARD_MAX_BYTES", str(6 * 1024 * 1024)))   # ~6 MiB
# "Work" budget per shard; bigger matrices consume more budget (see estimate_cost)
DEFAULT_COST_BUDGET = float(os.getenv("SHARD_COST_BUDGET", "2.0e7"))
# ------------------------------------------------------------------

READ_BUF_MB    = 16
TEMP_CONTAINER = "temp"

def _bsc():
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _iter_text(bc):
    """Stream a blob as text with bounded memory."""
    chunks = bc.download_blob(max_concurrency=2).chunks()
    class _Raw(io.RawIOBase):
        def __init__(self, it): self.it, self.buf = iter(it), b""
        def readable(self): return True
        def readinto(self, b):
            if not self.buf:
                try: self.buf = next(self.it)
                except StopIteration: return 0
            n = min(len(b), len(self.buf))
            b[:n] = self.buf[:n]
            self.buf = self.buf[n:]
            return n
    return io.TextIOWrapper(
        io.BufferedReader(_Raw(chunks), READ_BUF_MB * 1024 * 1024),
        encoding="utf-8"
    )

def estimate_cost(pair) -> float:
    """
    Very rough compute estimate so we keep shard work under a budget.
    Uses Strassen-ish complexity ~ n^2.807 (but simple and cheap to compute).
    """
    try:
        A = pair.get("A")
        n = len(A) if isinstance(A, list) else 64
        if n <= 0: n = 64
    except Exception:
        n = 64
    # n ** 2.807 ~ n^log2(7); scale down so values are in a usable range.
    return float(n ** 2.807)

def main(inBlob: func.InputStream, queueOut: func.Out[str]):
    """
    Blob-triggered sharder:
      - Streams a JSON array of {"A":..., "B":...} pairs
      - Writes shards to temp/runs/<run_id>/shards/shard-xxxxx.ndjson
      - Emits one matrix-shards queue message per shard
      - Writes temp/runs/<run_id>/manifest.json
    """
    # Effective thresholds (env overrides supported)
    MAX_LINES   = DEFAULT_MAX_LINES
    MAX_BYTES   = DEFAULT_MAX_BYTES
    COST_BUDGET = DEFAULT_COST_BUDGET

    # Create a unique run_id tied to input filename for easy tracing
    base = os.path.splitext(os.path.basename(inBlob.name))[0]
    run_id = f"{base}-{uuid.uuid4().hex[:8]}"

    # Stream input
    in_container, in_name = inBlob.name.split("/", 1)
    bc_in = _bsc().get_blob_client(in_container, in_name)
    txt = _iter_text(bc_in)

    prefix = f"runs/{run_id}/shards"
    shard_idx = 0

    buf_lines = []           # NDJSON lines waiting to flush
    buf_bytes = 0            # current buffer size (bytes)
    cost_used = 0.0          # current shard's total "work" estimate
    total_pairs = 0          # overall count

    def flush():
        nonlocal shard_idx, buf_lines, buf_bytes, cost_used
        if not buf_lines:
            return
        shard_idx += 1
        shard_name = f"{prefix}/shard-{shard_idx:05d}.ndjson"
        data = ("\n".join(buf_lines) + "\n").encode("utf-8")

        _bsc().get_blob_client(TEMP_CONTAINER, shard_name).upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/x-ndjson"),
        )

        # one queue message per shard
        queueOut.set(json.dumps({
            "run_id": run_id,
            "shard_no": shard_idx,
            "shard_blob": f"{TEMP_CONTAINER}/{shard_name}"
        }))

        logging.info(
            "Shard %d written: %d rows, %.2f MiB, est_cost=%.2e",
            shard_idx, len(buf_lines), len(data) / (1024 * 1024), cost_used
        )
        # reset accumulators
        buf_lines.clear()
        buf_bytes = 0
        cost_used = 0.0

    # Read the top-level array: [ {...}, {...}, ... ]
    for pair in ijson.items(txt, "item"):
        # Keep the original JSON for the worker so it doesn’t parse twice
        line = json.dumps(pair, separators=(",", ":"))
        size = len(line) + 1  # + newline
        cst  = estimate_cost(pair)

        # If this single row is too big, flush any current and write it alone
        if (buf_lines and (
            len(buf_lines) + 1 > MAX_LINES or
            buf_bytes + size > MAX_BYTES or
            cost_used + cst > COST_BUDGET
        )):
            flush()

        buf_lines.append(line)
        buf_bytes += size
        cost_used += cst
        total_pairs += 1

        # If we somehow exceed thresholds after adding, flush immediately
        if (len(buf_lines) >= MAX_LINES or
            buf_bytes >= MAX_BYTES or
            cost_used >= COST_BUDGET):
            flush()

    # Flush any tail
    flush()

    # Write manifest for the merger
    manifest = {
        "run_id": run_id,
        "expected_parts": shard_idx,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "max_lines": MAX_LINES,
        "max_bytes": MAX_BYTES,
        "cost_budget": COST_BUDGET,
        "total_pairs": total_pairs,
    }
    _bsc().get_blob_client(TEMP_CONTAINER, f"runs/{run_id}/manifest.json").upload_blob(
        json.dumps(manifest).encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

    logging.info(
        "Split complete: run=%s shards=%d total_pairs=%d (max_lines=%d, max_bytes=%d, cost_budget=%.2e)",
        run_id, shard_idx, total_pairs, MAX_LINES, MAX_BYTES, COST_BUDGET
    )
