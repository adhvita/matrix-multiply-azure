import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
import os, io, time, json, logging
import numpy as np
import uuid, pathlib
# CHANGE: also import next_pow2 for the pad-ratio guard
from shared.strassen_module import strassen_rectangular, next_pow2

# App settings (can override in Azure)
OUTPUT_CONTAINER    = os.getenv("OUTPUT_CONTAINER", "output-container")
DEFAULT_DTYPE       = os.getenv("MM_DTYPE", "float32").lower()     # float32/float64
STRASSEN_THRESHOLD  = int(os.getenv("STRASSEN_THRESHOLD", "1024")) # high crossover
MAX_DIM_SINGLE      = int(os.getenv("MAX_DIM_SINGLE", "6144"))     # route big to Durable
TILE_SIZE           = int(os.getenv("TILE_SIZE", "2048"))          # Durable tile size
TEMP_CONTAINER      = os.getenv("TEMP_CONTAINER", "temp")          # tiles/partials
PAD_RATIO_LIMIT     = float(os.getenv("PAD_RATIO_LIMIT", "1.5"))
RUN_LOG_CONTAINER   = os.getenv("RUN_LOG_CONTAINER", OUTPUT_CONTAINER)
RUN_LOG_PREFIX      = os.getenv("RUN_LOG_PREFIX", "runs/")
def jlog(payload: dict):
    line = json.dumps(payload, ensure_ascii=False)
    logging.getLogger("router").info(line)  # App Insights

    # append to blob: <RUN_LOG_CONTAINER>/<RUN_LOG_PREFIX>run_<run_id>.jsonl
    try:
        run_id = payload.get("run_id", "unknown")
        cc = _blob_client().get_container_client(RUN_LOG_CONTAINER)
        blob_name = f"{RUN_LOG_PREFIX}run_{run_id}.jsonl"
        _append_blob_line(cc, blob_name, line)
    except Exception as e:
        logging.getLogger("router").warning(f"blob-append-log failed: {e}")

def _logger():
    lg = logging.getLogger("router")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    # behave on Consumption: avoid BLAS oversubscription
    for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(v, "1")
    return lg

def _blob_client():
    return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

def _upload_npy(cc, name: str, arr: np.ndarray):
    buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
    cc.upload_blob(name, buf.getvalue(), overwrite=True,
                   content_settings=ContentSettings(content_type="application/octet-stream"))

def _append_blob_line(cc, name: str, text: str):
    from azure.core.exceptions import ResourceExistsError
    bc = cc.get_blob_client(name)
    try:
        bc.create_append_blob()
    except ResourceExistsError:
        pass
    bc.append_block((text + "\n").encode("utf-8"))


async def main(inputBlob: func.InputStream, starter: str):
    # Durable client binding
    import azure.durable_functions as df
    client = df.DurableOrchestrationClient(starter)
    logger = _logger()

    name = inputBlob.name  # e.g., inputs/pair_....npz
    logger.info(f"Triggered by blob: {name} size={inputBlob.length} bytes")

    data = inputBlob.read()
    if not name.lower().endswith(".npz"):
        logger.error("Expected .npz with keys A,B")
        return

    # Safe: npz is non-pickled by default
    npz = np.load(io.BytesIO(data))
    if not ("A" in npz and "B" in npz):
        logger.error("NPZ missing keys 'A' and 'B'")
        return

    A = np.array(npz["A"], copy=False)
    B = np.array(npz["B"], copy=False)

    # Validate
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        logger.error(f"Invalid shapes. Expected square NxN; got A{A.shape}, B{B.shape}")
        return

    N = int(A.shape[0])
    try:
        base = name.split("/")[-1]
        run_id = base[:-4] if base.lower().endswith(".npz") else base
    except Exception:
        run_id = str(uuid.uuid4())      

    # CHANGE: compute pad ratio *after* we know N
    P = next_pow2(N)
    pad_ratio = P / float(N)
    common_ctx = {
    "ts": time.time(),
    "run_id": run_id,
    "N": N,
    "dtype": str(A.dtype),
    "threshold": STRASSEN_THRESHOLD,
    "tile": TILE_SIZE,
    "pad_ratio": pad_ratio
    }
    target_dtype = np.float32 if DEFAULT_DTYPE == "float32" else np.float64
    if A.dtype != target_dtype:
        logger.info(f"Coercing A from {A.dtype} -> {target_dtype}")
        A = A.astype(target_dtype, copy=False)
    if B.dtype != target_dtype:
        logger.info(f"Coercing B from {B.dtype} -> {target_dtype}")
        B = B.astype(target_dtype, copy=False)

    logger.info(
        f"N={N}, dtype={A.dtype}, threshold={STRASSEN_THRESHOLD}, "
        f"MAX_DIM_SINGLE={MAX_DIM_SINGLE}, pad_ratio={pad_ratio:.3f} (P={P})"
    )

    bsc = _blob_client()
    out_cc = bsc.get_container_client(OUTPUT_CONTAINER)

    # CHANGE: route to Durable if size is big OR padding would be excessive
    if N > MAX_DIM_SINGLE or pad_ratio > PAD_RATIO_LIMIT:
        reason = "N>MAX_DIM_SINGLE" if N > MAX_DIM_SINGLE else f"pad_ratio>{PAD_RATIO_LIMIT}"
        logger.info(f"Routing to Durable ({reason}).")

        instance_id = await client.start_new("orchestrator", None, {
            "run_id": run_id,
            "input_container": "inputs",
            "input_blob": name.split("/", 1)[-1],
            "temp_container": TEMP_CONTAINER,
            "output_container": OUTPUT_CONTAINER,
            "tile_size": TILE_SIZE,
            "dtype": "float32" if target_dtype==np.float32 else "float64",
            "strassen_threshold": STRASSEN_THRESHOLD
        })
        logger.info(f"Started durable instance: {instance_id}")
        instance_id = await client.start_new("orchestrator", None, orchestrator_input)

        payload = dict(common_ctx)
        payload.update({
            "mode": "durable_start",
            "instance_id": instance_id,
            "output": None
        })
        jlog(payload)                      # local JSONL
        logger.info(json.dumps(payload))   # cloud trace

        logger.info(f"Started durable instance: {instance_id}")
        return

    # Inline (single-invocation) path
    logger.info("Routing to inline (single invocation).")
    t0 = time.time()
    C = strassen_rectangular(A, B, threshold=STRASSEN_THRESHOLD, logger=logger)
    t1 = time.time()

    out_blob = f"C_{N}x{N}_{'float32' if target_dtype==np.float32 else 'float64'}_{t1}.npy"
    _upload_npy(out_cc, out_blob, C)
    
    payload = dict(common_ctx)
    payload.update({
        "mode": "inline",
        "compute_sec": round(t1 - t0, 6),
        "output": f"{OUTPUT_CONTAINER}/{out_blob}"
    })
    jlog(payload)
    logger.info(json.dumps(payload))
    return

    # logger.info(json.dumps({
    #     "mode": "inline",
    #     "N": N,
    #     "dtype": str(A.dtype),
    #     "threshold": STRASSEN_THRESHOLD,
    #     "pad_ratio": round(pad_ratio, 6),
    #     "compute_sec": round(t1 - t0, 6),
    #     "output": f"{OUTPUT_CONTAINER}/{out_blob}"
    # }))

# import azure.functions as func
# from azure.storage.blob import BlobServiceClient, ContentSettings
# import os, io, time, json, logging
# import numpy as np

# from shared.strassen_module import strassen_rectangular

# # App settings (can override in Azure)
# OUTPUT_CONTAINER    = os.getenv("OUTPUT_CONTAINER", "output-container")
# DEFAULT_DTYPE       = os.getenv("MM_DTYPE", "float32").lower()     # float32/float64
# STRASSEN_THRESHOLD  = int(os.getenv("STRASSEN_THRESHOLD", "1024")) # high crossover
# MAX_DIM_SINGLE      = int(os.getenv("MAX_DIM_SINGLE", "6144"))     # route big to Durable
# TILE_SIZE           = int(os.getenv("TILE_SIZE", "2048"))          # Durable tile size
# TEMP_CONTAINER      = os.getenv("TEMP_CONTAINER", "temp")          # tiles/partials

# from shared.strassen_module import next_pow2  # or wherever you defined it
# PAD_RATIO_LIMIT = float(os.getenv("PAD_RATIO_LIMIT", "1.5"))
# P = next_pow2(N)
# pad_ratio = P / float(N)

# def _logger():
#     lg = logging.getLogger("router")
#     if not lg.handlers:
#         h = logging.StreamHandler()
#         h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
#         lg.addHandler(h)
#         lg.setLevel(logging.INFO)
#     # behave on Consumption: avoid BLAS oversubscription
#     for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
#         os.environ.setdefault(v, "1")
#     return lg

# def _blob_client():
#     return BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

# def _upload_npy(cc, name: str, arr: np.ndarray):
#     buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
#     cc.upload_blob(name, buf.getvalue(), overwrite=True,
#                    content_settings=ContentSettings(content_type="application/octet-stream"))

# def main(inputBlob: func.InputStream, starter: str):
#     # Durable client binding
#     import azure.durable_functions as df
#     client = df.DurableOrchestrationClient(starter)
#     logger = _logger()

#     name = inputBlob.name  # e.g., inputs/pair_....npz
#     logger.info(f"Triggered by blob: {name} size={inputBlob.length} bytes")

#     data = inputBlob.read()
#     if not name.lower().endswith(".npz"):
#         logger.error("Expected .npz with keys A,B")
#         return

#     npz = np.load(io.BytesIO(data))
#     if not ("A" in npz and "B" in npz):
#         logger.error("NPZ missing keys 'A' and 'B'")
#         return

#     A = np.array(npz["A"], copy=False)
#     B = np.array(npz["B"], copy=False)

#     # Validate
#     if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
#         logger.error(f"Invalid shapes. Expected square NxN; got A{A.shape}, B{B.shape}")
#         return

#     N = int(A.shape[0])
#     target_dtype = np.float32 if DEFAULT_DTYPE == "float32" else np.float64
#     if A.dtype != target_dtype: A = A.astype(target_dtype, copy=False)
#     if B.dtype != target_dtype: B = B.astype(target_dtype, copy=False)

#     logger.info(f"N={N}, dtype={A.dtype}, threshold={STRASSEN_THRESHOLD}, MAX_DIM_SINGLE={MAX_DIM_SINGLE}")

#     bsc = _blob_client()
#     out_cc = bsc.get_container_client(OUTPUT_CONTAINER)

#     if N <= MAX_DIM_SINGLE:
#         # Inline (single-invocation) path
#         t0 = time.time()
#         C = strassen_rectangular(A, B, threshold=STRASSEN_THRESHOLD, logger=logger)
#         t1 = time.time()
#         out_blob = f"C_{N}x{N}_{'float32' if target_dtype==np.float32 else 'float64'}.npy"
#         _upload_npy(out_cc, out_blob, C)
#         logger.info(json.dumps({
#             "mode": "inline",
#             "N": N,
#             "dtype": str(A.dtype),
#             "threshold": STRASSEN_THRESHOLD,
#             "compute_sec": round(t1 - t0, 6),
#             "output": f"{OUTPUT_CONTAINER}/{out_blob}"
#         }))
#         return

#     # Durable path for large N
#     instance_id = client.start_new("orchestrator", None, {
#         "input_container": "inputs",
#         "input_blob": name.split("/", 1)[-1],
#         "temp_container": TEMP_CONTAINER,
#         "output_container": OUTPUT_CONTAINER,
#         "tile_size": TILE_SIZE,
#         "dtype": "float32" if target_dtype==np.float32 else "float64",
#         "strassen_threshold": STRASSEN_THRESHOLD
#     })
#     logger.info(f"Started durable instance: {instance_id}")
