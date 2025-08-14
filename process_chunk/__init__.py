import os, io, json, logging
import azure.functions as func
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings

# import your existing Strassen
from ..strassen_algo.strassen_module import strassen  

JSONL_PREVIEW = 8
READ_BUF_MB   = 8
TEMP_CONTAINER = "temp"

def _bsc():
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

#def _download_lines(container, name):
    bc = _bsc().get_blob_client(container, name)
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
    f = io.TextIOWrapper(io.BufferedReader(_Raw(chunks), READ_BUF_MB*1024*1024), encoding="utf-8")
    for line in f:
        line = line.strip()
        if line: yield line

#def _next_pow2(n): return 1 if n<=1 else 1<<(n-1).bit_length()
#def _pad_to_pow2(M, k):
    if M.shape[0]==k: return M
    P=np.zeros((k,k), dtype=M.dtype); P[:M.shape[0], :M.shape[1]]=M; return P
#def _strassen_padded(A,B):
    n=A.shape[0]; k=_next_pow2(n)
    Ap=_pad_to_pow2(A,k); Bp=_pad_to_pow2(B,k)
    Cp=strassen(Ap,Bp); return Cp[:n,:n]

#def main(msg: func.QueueMessage, doneOut: func.Out[str]):
    payload = json.loads(msg.get_body().decode())
    run_id    = payload["run_id"]
    shard_no  = int(payload["shard_no"])
    shard_ref = payload["shard_blob"]  # "temp/runs/<run_id>/shards/shard-00001.ndjson"
    in_container, in_name = shard_ref.split("/", 1)

    # output part path: temp/runs/<run_id>/parts/part-00001.jsonl
    out_name = f"runs/{run_id}/parts/part-{shard_no:05d}.jsonl"
    bc_out = _bsc().get_blob_client(TEMP_CONTAINER, out_name)
    try:
        bc_out.create_append_blob(content_settings=ContentSettings(content_type="application/json"))
    except Exception:
        pass

    BUF = bytearray()
    TARGET = 256*1024

    def flush():
        nonlocal BUF
        if not BUF: return
        bc_out.append_block(bytes(BUF))
        BUF.clear()

    # If your input includes an 'index' field, we’ll preserve it; else we’ll just emit sequential order
    seq = 0
    for line in _download_lines(in_container, in_name):
        pair = json.loads(line)
        A = np.asarray(pair["A"], dtype=np.float32)
        B = np.asarray(pair["B"], dtype=np.float32)
        n = A.shape[0]
        C = _strassen_padded(A,B)

        rec = {
            "index": pair.get("index", None),    # keep index if present
            "shape": [int(n), int(n)],
            "sum": float(np.sum(C)),
            "mean": float(np.mean(C)),
            "A_preview": (A[:JSONL_PREVIEW,:JSONL_PREVIEW].tolist() if n>JSONL_PREVIEW else A.tolist()),
            "B_preview": (B[:JSONL_PREVIEW,:JSONL_PREVIEW].tolist() if n>JSONL_PREVIEW else B.tolist()),
            "C_preview": (C[:JSONL_PREVIEW,:JSONL_PREVIEW].tolist() if n>JSONL_PREVIEW else C.tolist())
        }
        seq += 1

        BUF.extend((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))
        if len(BUF) >= TARGET: flush()
    flush()

    doneOut.set(json.dumps({ "run_id": run_id, "part_done": shard_no }))
    logging.info("Processed shard %d run %s -> %s/%s", shard_no, run_id, TEMP_CONTAINER, out_name)

def main(msg: func.QueueMessage, mergeOut: func.Out[str]):
    """
    Queue trigger: processes one shard NDJSON and writes one parts file.
    Finally posts a merge message for the run.
    """
    try:
        payload = json.loads(msg.get_body().decode("utf-8"))
        run_id     = payload["run_id"]
        shard_no   = int(payload["shard_no"])
        shard_blob = payload["shard_blob"]  # "temp/runs/<run_id>/shards/shard-00001.ndjson"

        logging.info("process_chunk: run=%s shard_no=%d shard_blob=%s", run_id, shard_no, shard_blob)

        # Parse "temp/<path>" into (container, name)
        container, name = shard_blob.split("/", 1) if "/" in shard_blob else (TEMP_CONTAINER, shard_blob)

        bsc = _bsc()
        bc_in = bsc.get_blob_client(container, name)

        # Stream NDJSON lines:
        downloader = bc_in.download_blob(max_concurrency=2)
        stream = downloader.chunks()

        # Prepare parts output
        parts_prefix = f"runs/{run_id}/parts"
        part_name = f"{parts_prefix}/part-{shard_no:05d}.jsonl"
        bc_out = bsc.get_blob_client(TEMP_CONTAINER, part_name)

        # Create/overwrite a small block blob for the part
        bc_out.upload_blob(b"", overwrite=True, content_settings=ContentSettings(content_type="application/x-ndjson"))

        # Lazy import to avoid heavy module init at worker start
        import numpy as np
        from ..strassen_algo.strassen_module import strassen  # adjust if path differs

        # Process each NDJSON record (one pair per line)
        buf = bytearray()
        count = 0
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

                try:
                    pair = json.loads(line)
                    A = np.array(pair["A"], dtype=np.float32)
                    B = np.array(pair["B"], dtype=np.float32)
                    # strassen can be heavy; keep sizes reasonable by sharding upstream
                    C = strassen(A, B)
                    # Emit a compact summary row per pair
                    out_row = {
                        "index": count,
                        "shape": [int(A.shape[0]), int(A.shape[1])],
                        "sum": float(np.sum(C)),
                        "mean": float(np.mean(C))
                    }
                except Exception as ex:
                    out_row = {"index": count, "error": str(ex)}

                bc_out.append_block((json.dumps(out_row, separators=(",", ":")) + "\n").encode("utf-8"))
                count += 1

        # Finalize: tell merger we have (at least) this part ready.
        mergeOut.set(json.dumps({
            "run_id": run_id,
            # optional: provide expected_parts if you want; merger can also read manifest.json
            "ts": datetime.now(timezone.utc).isoformat()
        }))

        logging.info("process_chunk: done run=%s shard_no=%d wrote %s (records=%d)", run_id, shard_no, part_name, count)

    except Exception:
        logging.exception("process_chunk failed")
        # Let Functions move to poison after maxDequeueCount
        raise