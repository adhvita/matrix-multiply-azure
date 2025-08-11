import os, io, json, time, logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
try:
    # Newer SDKs
    from azure.storage.blob import BlobRequestConditions       # type: ignore
    _HAVE_COND = True
except Exception:
    _HAVE_COND = False
from azure.core.exceptions import ResourceExistsError
import ijson, numpy as np

from ..strassen_algo.strassen_module import strassen  # keep your Strassen

MAX_RUN_SECONDS = 120
JSONL_PREVIEW   = 8

def _next_pow2(n:int)->int: return 1 if n<=1 else 1<<(n-1).bit_length()
def _pad_to_pow2(M, size):
    if M.shape[0]==size: return M
    P = np.zeros((size,size), dtype=M.dtype); P[:M.shape[0], :M.shape[1]] = M; return P
def _strassen_padded(A,B):
    n=A.shape[0]; m=_next_pow2(n)
    Ap=_pad_to_pow2(A,m); Bp=_pad_to_pow2(B,m)
    Cp=strassen(Ap,Bp); return Cp[:n,:n]

class _IterStream(io.RawIOBase):
    def __init__(self,it): self.it, self.buf = iter(it), b""
    def readable(self): return True
    def readinto(self,b):
        if not self.buf:
            try: self.buf = next(self.it)
            except StopIteration: return 0
        n=min(len(b), len(self.buf)); b[:n]=self.buf[:n]; self.buf=self.buf[n:]; return n

def _download_text_stream(conn, container, name):
    bsc=BlobServiceClient.from_connection_string(conn)
    bc=bsc.get_blob_client(container=container, blob=name)
    raw=bc.download_blob(max_concurrency=1).chunks()
    return io.TextIOWrapper(io.BufferedReader(_IterStream(raw), 8*1024*1024), encoding="utf-8")

def _iter_pairs(txt): 
    for pair in ijson.items(txt, "item"): yield pair

def _get_append_client(conn, full_path:str):
    # full_path like "output-container/result-foo.jsonl"
    container, blob = full_path.split("/",1)
    bsc=BlobServiceClient.from_connection_string(conn)
    bc=bsc.get_blob_client(container=container, blob=blob)
    try:
        bc.create_append_blob(content_settings=ContentSettings(content_type="application/json"))
    except ResourceExistsError:
        pass
    return bc

def _append_line(bc, line:str):
    data = (line if line.endswith("\n") else line + "\n").encode("utf-8")

    if _HAVE_COND:
        # Idempotent append when the type is available
        props = bc.get_blob_properties()
        pos = props.size
        cond = BlobRequestConditions()
        cond.if_append_position_equal = pos
        try:
            bc.append_block(data, conditions=cond)
        except Exception as e:
            logging.warning("Skip duplicate append (position changed): %s", e)
    else:
        # Older SDK: just append (may duplicate on retries)
        bc.append_block(data)

def main(msg: func.QueueMessage, queueOut: func.Out[str]):
    t0=time.time()
    payload=json.loads(msg.get_body().decode())

    blob_full   = payload["blob"]
    start_index = int(payload.get("start_index",0))
    chunk_size  = int(payload.get("chunk_size",200))
    out_blob    = payload.get("out_blob")

    # Fallback for older messages without out_blob
    if not out_blob:
        _, in_name = blob_full.split("/",1)
        out_blob = f"output-container/result-{in_name.replace('/', '_')}.jsonl"

    in_container, in_name = blob_full.split("/",1)
    conn=os.environ["AzureWebJobsStorage"]
    logging.info("Chunk: %s from index=%d (chunk=%d) -> %s", blob_full, start_index, chunk_size, out_blob)

    out_client=_get_append_client(conn, out_blob)
    txt=_download_text_stream(conn, in_container, in_name)

    processed=0; idx=0
    for pair in _iter_pairs(txt):
        if idx < start_index: idx += 1; continue

        ok=True; err=None
        try:
            A=np.array(pair.get("A"), dtype=np.float32, copy=False)
            B=np.array(pair.get("B"), dtype=np.float32, copy=False)
            if A.ndim!=2 or B.ndim!=2 or A.shape!=B.shape or A.shape[0]!=A.shape[1]:
                ok=False; err=f"invalid shapes {A.shape} vs {B.shape}"
        except Exception as e:
            ok=False; err=f"parse error: {e}"

        if ok:
            n=A.shape[0]
            C=_strassen_padded(A,B)
            line={
                "index": idx,
                "shape":[int(n),int(n)],
                "sum": float(np.sum(C)),
                "mean": float(np.mean(C)),
                "A_preview": (A[:JSONL_PREVIEW,:JSONL_PREVIEW].tolist() if n>JSONL_PREVIEW else A.tolist()),
                "B_preview": (B[:JSONL_PREVIEW,:JSONL_PREVIEW].tolist() if n>JSONL_PREVIEW else B.tolist()),
                "C_preview": (C[:JSONL_PREVIEW,:JSONL_PREVIEW].tolist() if n>JSONL_PREVIEW else C.tolist())
            }
            del A,B,C
        else:
            line={"index": idx, "error": err}

        _append_line(out_client, json.dumps(line))

        idx += 1; processed += 1
        if processed >= chunk_size or (time.time()-t0) > MAX_RUN_SECONDS:
            queueOut.set(json.dumps({
                "blob": blob_full,
                "start_index": idx,
                "chunk_size": chunk_size,
                "out_blob": out_blob
            }))
            logging.info("Re-queued continuation at index %d", idx)
            return

    logging.info("Completed %s at last index %d → %s", blob_full, idx-1, out_blob)
