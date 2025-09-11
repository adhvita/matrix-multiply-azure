import logging, os, io, json, math, pathlib, uuid, time
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError
from shared.strassen_module import strassen_rectangular

# ---------- helpers ----------
def _download_blob_bytes(cc, name: str):
    data = cc.get_blob_client(name).download_blob().readall()
    return data, len(data)

def _upload_npy_with_size(cc, name: str, arr: np.ndarray) -> int:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    data = buf.getvalue()
    cc.upload_blob(
        name, data, overwrite=True,
        content_settings=ContentSettings(content_type="application/octet-stream"),
    )
    return len(data)

def _safe_run_dir():
    p = pathlib.Path(os.getenv("LOCAL_RUN_LOG_DIR") or
                     ("/temp/runs" if os.getenv("WEBSITE_INSTANCE_ID") else "./runs"))
    try: p.mkdir(parents=True, exist_ok=True)
    except Exception: pass
    return p

RUN_LOG_DIR = _safe_run_dir()

def _append_blob_line(cc, name: str, text: str):
    bc = cc.get_blob_client(name)
    try: bc.create_append_blob()
    except ResourceExistsError: pass
    bc.append_block((text + "\n").encode("utf-8"))

def jlog(payload: dict, fname: str = "local_metrics.jsonl"):
    line = json.dumps(payload, ensure_ascii=False)
    logging.getLogger("activity").info(line)               # App Insights
    try:
        with (RUN_LOG_DIR / fname).open("a", encoding="utf-8") as f:
            f.write(line + "\n")                           # local fallback
    except Exception:
        pass
    cont = os.getenv("RUN_LOG_CONTAINER")
    pref = os.getenv("RUN_LOG_PREFIX", "runs/")
    if cont:
        try:
            cc = _bsc().get_container_client(cont)
            run_id = payload.get("run_id", "unknown")
            _append_blob_line(cc, f"{pref}run_{run_id}.jsonl", line)
        except Exception as e:
            logging.getLogger("activity").warning(f"blob-append-log failed: {e}")

def _logger():
    lg = logging.getLogger("activity")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(v, "1")
    return lg

def _bsc():
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _load_npy_from_blob(cc, name, allow_pickle=False):
    data = cc.get_blob_client(name).download_blob().readall()
    return np.load(io.BytesIO(data), allow_pickle=allow_pickle)

def _dtype_of(s: str):
    return np.float32 if s == "float32" else np.float64

# ---------- main ----------
def main(payload: dict) -> dict:
    logger = _logger()
    op = payload.get("op")
    bsc = _bsc()
    run_id = payload.get("run_id") or f"local-{uuid.uuid4()}"

    if op == "prepare_tiles":
        icc = bsc.get_container_client(payload["input_container"])
        tcc = bsc.get_container_client(payload["temp_container"])
        tile = int(payload["tile"])
        dt = _dtype_of(payload["dtype"])
        dtype_str = "float32" if dt == np.float32 else "float64"

        # FIX: single download (count + load from same bytes)
        t0 = time.time(); bytes_in = 0; bytes_out = 0
        raw, nbytes = _download_blob_bytes(icc, payload["input_blob"])
        bytes_in += nbytes
        npz = np.load(io.BytesIO(raw))                     # reuse raw bytes

        if not ("A" in npz and "B" in npz): raise ValueError("NPZ missing A,B")
        A = np.array(npz["A"], copy=False); B = np.array(npz["B"], copy=False)
        if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Invalid shapes: A{A.shape} B{B.shape}")

        N_content = int(A.shape[0])
        if "N" in payload and int(payload["N"]) != N_content:
            logger.warning(f"prepare_tiles: payload N={payload['N']} != content N={N_content}; using content")
        N = N_content

        if A.dtype != dt: A = A.astype(dt, copy=False)
        if B.dtype != dt: B = B.astype(dt, copy=False)

        tiles = math.ceil(N / tile)
        logger.info(f"prepare_tiles: N={N}, dtype={dtype_str}, tile={tile}, tiles_per_side={tiles}")

        # write tiles (count bytes_out)
        for i in range(tiles):
            r0 = i*tile; r1 = min((i+1)*tile, N)
            for q in range(tiles):
                c0 = q*tile; c1 = min((q+1)*tile, N)
                T = np.zeros((tile, tile), dtype=dt)
                T[:(r1-r0), :(c1-c0)] = A[r0:r1, c0:c1]
                bytes_out += _upload_npy_with_size(tcc, f"A_{i}_{q}.npy", T)

        for q in range(tiles):
            r0 = q*tile; r1 = min((q+1)*tile, N)
            for j in range(tiles):
                c0 = j*tile; c1 = min((j+1)*tile, N)
                T = np.zeros((tile, tile), dtype=dt)
                T[:(r1-r0), :(c1-c0)] = B[r0:r1, c0:c1]
                bytes_out += _upload_npy_with_size(tcc, f"B_{q}_{j}.npy", T)

        # manifest (optional)
        bytes_out += _upload_npy_with_size(tcc, "manifest.npy",
                                           np.array([N, tile, tiles], dtype=np.int64))

        t1 = time.time()
        rec = {
            "ts": time.time(), "run_id": run_id, "op": "prepare_tiles",
            "N": N, "tile": tile, "tiles": tiles, "dtype": dtype_str,
            "bytes_in": int(bytes_in), "bytes_out": int(bytes_out),
            "dur_ms": int((t1 - t0) * 1000)
        }
        jlog(rec, fname=f"run_{run_id}.jsonl")
        return {"N": N, "tile": tile, "tiles": tiles}

    elif op == "multiply_tile_rowcol":
        tcc = bsc.get_container_client(payload["temp_container"])
        i = int(payload["i"]); j = int(payload["j"])
        tiles = int(payload["tiles"]); tile = int(payload["tile"])
        dt = _dtype_of(payload["dtype"]); dtype_str = "float32" if dt==np.float32 else "float64"
        thr = int(payload["strassen_threshold"])

        t0 = time.time(); bytes_in = 0; bytes_out = 0
        partials = []
        for q in range(tiles):
            Aiq = _load_npy_from_blob(tcc, f"A_{i}_{q}.npy")
            Bqj = _load_npy_from_blob(tcc, f"B_{q}_{j}.npy")
            bytes_in += int(Aiq.nbytes + Bqj.nbytes)       # FIX: count reads

            if tile <= thr:
                Cpart = Aiq.dot(Bqj)
            else:
                class Dummy: pass
                dummy = Dummy(); dummy.debug = dummy.info = lambda *a, **k: None
                Cpart = strassen_rectangular(Aiq, Bqj, threshold=thr, logger=dummy)

            p = f"partials/part_{i}_{q}_{j}.npy"
            bytes_out += _upload_npy_with_size(tcc, p, Cpart.astype(dt, copy=False))  # FIX: single uploader
            partials.append(p)                                                        # FIX: append once

        t1 = time.time()
        rec = {
            "ts": time.time(), "run_id": run_id, "op": "multiply_tile_rowcol",
            "N": int(payload.get("N", tiles * tile)), "tile": tile, "i": i, "j": j, "dtype": dtype_str,
            "bytes_in": int(bytes_in), "bytes_out": int(bytes_out),
            "dur_ms": int((t1 - t0) * 1000)
        }
        jlog(rec, fname=f"run_{run_id}.jsonl")
        return {"i": i, "j": j, "partials": partials}

    elif op == "reduce_partials":
        tcc = bsc.get_container_client(payload["temp_container"])
        i = int(payload["i"]); j = int(payload["j"])
        tile = int(payload["tile"])
        dt = _dtype_of(payload["dtype"]); dtype_str = "float32" if dt==np.float32 else "float64"

        t0 = time.time(); bytes_in = 0; bytes_out = 0
        acc = None
        for p in payload["partials"]:
            Cpart = _load_npy_from_blob(tcc, p)
            bytes_in += int(Cpart.nbytes)                 # FIX: count reads
            acc = Cpart if acc is None else (acc + Cpart)

        out_name = f"C_{i}_{j}.npy"
        bytes_out += _upload_npy_with_size(tcc, out_name, acc.astype(dt, copy=False)) # FIX: single uploader

        t1 = time.time()
        rec = {
            "ts": time.time(), "run_id": run_id, "op": "reduce_partials",
            "N": int(payload.get("N", 0)), "tile": tile, "i": i, "j": j, "dtype": dtype_str,
            "bytes_in": int(bytes_in), "bytes_out": int(bytes_out),
            "dur_ms": int((t1 - t0) * 1000)
        }
        jlog(rec, fname=f"run_{run_id}.jsonl")
        return {"i": i, "j": j, "tile": tile, "name": out_name}

    elif op == "merge_tiles":
        tcc = bsc.get_container_client(payload["temp_container"])
        occ = bsc.get_container_client(payload["output_container"])
        N = int(payload["N"]); tile = int(payload["tile"]); tiles = int(payload["tiles"])
        dt = _dtype_of(payload["dtype"]); dtype_str = "float32" if dt==np.float32 else "float64"

        t0 = time.time(); bytes_in = 0; bytes_out = 0
        C = np.zeros((N, N), dtype=dt)
        for i in range(tiles):
            r0 = i*tile; r1 = min((i+1)*tile, N)
            for j in range(tiles):
                c0 = j*tile; c1 = min((j+1)*tile, N)
                Tij = _load_npy_from_blob(tcc, f"C_{i}_{j}.npy")
                bytes_in += int(Tij.nbytes)               # FIX: count reads
                C[r0:r1, c0:c1] = Tij[:(r1-r0), :(c1-c0)]

        out_blob = f"C_{N}x{N}_{dtype_str}.npy"
        bytes_out += _upload_npy_with_size(occ, out_blob, C)  # FIX: single uploader

        t1 = time.time()
        # compact summary + durable_done
        rec = {
            "ts": time.time(), "run_id": run_id, "op": "merge_tiles",
            "N": N, "tile": tile, "tiles": tiles, "dtype": dtype_str,
            "bytes_in": int(bytes_in), "bytes_out": int(bytes_out),
            "dur_ms": int((t1 - t0) * 1000),
            "output": out_blob
        }
        jlog(rec, fname=f"run_{run_id}.jsonl")
        return out_blob

    else:
        raise ValueError(f"Unknown op: {op}")
