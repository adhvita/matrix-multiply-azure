import os, cv2, numpy as np, onnxruntime as ort, requests
import logging, time, shutil
from datetime import datetime, timedelta, timezone   # CHANGED: use timezone-aware
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

def cd(**k):  # custom dimensions helper
    return {'custom_dimensions': k}

CONF = float(os.getenv("CONF_THRESHOLD", "0.3"))
IOU  = float(os.getenv("IOU_NMS", "0.5"))
MODEL_URL = os.getenv("MODEL_URL", "")  # optional now
VIDEO_CONTAINER = os.getenv("VIDEO_CONTAINER", "videos-in")  # NEW

_session = None
_model_path = "/tmp/model.onnx"

# NEW: allow overriding packaged model name/location
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "models/yolo5.onnx")

def _pkg_model_candidates():
    """
    Return plausible locations of the packaged model inside /home/site/wwwroot.
    """
    here = os.path.dirname(os.path.abspath(__file__))                     # .../wwwroot/process_segment
    site = os.path.abspath(os.path.join(here, ".."))                      # .../wwwroot
    return [
        os.path.join(site, MODEL_LOCAL_PATH),                              # preferred (models/yolo5.onnx)
        os.path.join(site, "models", "yolo5.onnx"),                        # fallback name
    ]

# ---- SAS helpers ----
def _parse_conn():
    conn = os.getenv("AzureWebJobsStorage", "")
    parts = dict(p.split("=", 1) for p in conn.split(";") if "=" in p)
    return parts.get("AccountName"), parts.get("AccountKey")

def _make_blob_sas_url(container: str, name: str) -> str:
    acct, key = _parse_conn()
    if not acct or not key:
        raise RuntimeError("AzureWebJobsStorage account/key not found.")
    # CHANGED: timezone-aware datetimes (no utcnow)
    start  = datetime.now(timezone.utc) - timedelta(minutes=10)
    expiry = datetime.now(timezone.utc) + timedelta(hours=6)
    token = generate_blob_sas(
        account_name=acct,
        container_name=container,
        blob_name=name,
        account_key=key,
        permission=BlobSasPermissions(read=True),
        start=start,
        expiry=expiry,
    )
    return f"https://{acct}.blob.core.windows.net/{container}/{name}?{token}"

# ---- video opener ----
def _open_video(url: str, name: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap is not None and cap.isOpened():
        logging.info("video.open.ok", extra=cd(video=name, backend="opencv-ffmpeg"))
        return cap, "opencv-ffmpeg"

    logging.warning("video.open.try_imageio", extra=cd(video=name))
    try:
        import imageio.v3 as iio
        reader = iio.imiter(url, plugin="FFMPEG")  # streams http/https reliably
        class _CapIter:
            def __init__(self, it):
                self.it = it
                self.opened = True
            def isOpened(self): return self.opened
            def read(self):
                try:
                    frame_rgb = next(self.it)
                    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    return True, bgr
                except StopIteration:
                    self.opened = False
                    return False, None
            def release(self):
                self.opened = False
        cap2 = _CapIter(reader)
        logging.info("video.open.ok", extra=cd(video=name, backend="imageio-ffmpeg"))
        return cap2, "imageio-ffmpeg"
    except Exception as e:
        logging.error("video.open.failed", extra=cd(video=name, err=str(e)))
        return None, None

# ---- model/session helper ----
def _ensure_model():
    """
    Ensure /tmp/model.onnx exists. Prefer the packaged model; fall back to MODEL_URL.
    """
    global _session

    if _session is not None:
        return _session

    # If already staged in /tmp, use it.
    if not os.path.exists(_model_path):
        # CHANGED: prefer local packaged model (no network)
        staged = False
        for cand in _pkg_model_candidates():  # NEW
            if os.path.exists(cand):
                try:
                    os.makedirs(os.path.dirname(_model_path), exist_ok=True)
                    shutil.copyfile(cand, _model_path)
                    logging.info("model.staged.from_package", extra=cd(src=cand, dst=_model_path))
                    staged = True
                    break
                except Exception as e:
                    logging.warning("model.stage.failed", extra=cd(src=cand, err=str(e)))

        # If no packaged model, try network download as last resort (optional).
        if not staged:
            if MODEL_URL:
                logging.warning("model.download.begin", extra=cd(url="MODEL_URL"))
                r = requests.get(MODEL_URL, timeout=60)
                r.raise_for_status()
                with open(_model_path, "wb") as f:
                    f.write(r.content)
                logging.info("model.download.ok", extra=cd(size=len(r.content)))
            else:
                # No local and no URL – hard fail with a clear message.
                raise RuntimeError("Model not found in package and MODEL_URL not set.")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    _session = ort.InferenceSession(_model_path, sess_options=so, providers=["CPUExecutionProvider"])
    return _session

def _preprocess(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2; left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
    return x, scale, left, top

def _iou(b, others):
    xx1 = np.maximum(b[0], others[:, 0]); yy1 = np.maximum(b[1], others[:, 1])
    xx2 = np.minimum(b[2], others[:, 2]); yy2 = np.minimum(b[3], others[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area1 = (b[2] - b[0]) * (b[3] - b[1]); area2 = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(1e-6, area1 + area2 - inter)

def _nms(boxes, scores, iou=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        ious = _iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou]
    return keep

def _postprocess(pred, scale, left, top, conf=0.3, iou=0.5):
    p = pred[0]
    if p.ndim == 3: p = p[0]
    obj = p[:, 4:5]
    cls = p[:, 5:]
    cls_idx = np.argmax(cls, axis=1)
    cls_conf = cls[np.arange(cls.shape[0]), cls_idx]
    scores = (obj[:, 0] * cls_conf)
    mask = scores >= conf
    if not np.any(mask):
        return []
    p = p[mask]; scores = scores[mask]; cls_idx = cls_idx[mask]
    xy = p[:, 0:2]; wh = p[:, 2:4]
    x1y1 = xy - wh / 2; x2y2 = xy + wh / 2
    boxes = np.concatenate([x1y1, x2y2], axis=1)
    boxes[:, [0, 2]] -= left; boxes[:, [1, 3]] -= top
    boxes /= scale
    kept = _nms(boxes, scores, iou=iou)
    return [(boxes[i].tolist(), float(scores[i]), int(cls_idx[i])) for i in kept]

# ---- function entry ----
def main(seg: dict):
    sess = _ensure_model()

    name   = seg["name"]
    run_id = seg.get("runId", "")
    start  = float(seg["start"])
    end    = float(seg["end"])
    fps_target = float(seg.get("fps", 1.0))

    # CHANGED: mint a fresh SAS; if that fails, fall back to inbound seg["sas"]
    try:
        sas_url = _make_blob_sas_url(VIDEO_CONTAINER, name)
    except Exception as e:
        logging.warning("segment.sas.mint_failed", extra=cd(video=name, err=str(e)))
        sas_url = seg.get("sas", "")

    cap, backend = _open_video(sas_url, name)
    if not cap or not cap.isOpened():
        logging.error("segment.open_failed", extra=cd(runId=run_id, video=name))
        return []

    t0 = time.time()
    frames_done = 0
    next_emit = start
    out = []

    # Seek/time tracking per backend
    if backend == "opencv-ffmpeg":
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        def get_pos_s():
            return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    else:
        pos = start
        def get_pos_s():
            return pos

    while True:
        cur = get_pos_s()
        if cur >= end:
            break

        ok, frame = cap.read()
        if not ok:
            break

        if backend != "opencv-ffmpeg":
            pos += 1.0 / max(fps_target, 1e-6)

        if cur < next_emit:
            continue
        next_emit = cur + (1.0 / max(fps_target, 1e-6))

        x, scale, left, top = _preprocess(frame, 640)
        pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        dets = _postprocess(pred, scale, left, top, conf=CONF, iou=IOU)
        frames_done += 1
        for box, score, cls_id in dets:
            out.append({
                "t": round(cur, 3),
                "cls": cls_id,
                "score": round(score, 3),
                "box": [round(v, 1) for v in box]
            })

    try:
        cap.release()
    except Exception:
        pass

    dt = time.time() - t0
    logging.info("segment.done", extra=cd(
        runId     = run_id,
        video     = name,
        start     = start,
        seconds   = end - start,
        frames    = frames_done,
        duration_s= round(dt, 3)
    ))
    return out

# import os, cv2, numpy as np, onnxruntime as ort, requests
# import logging, time
# from datetime import datetime, timedelta
# from azure.storage.blob import generate_blob_sas, BlobSasPermissions

# def cd(**k):  # custom dimensions helper
#     return {'custom_dimensions': k}

# CONF = float(os.getenv("CONF_THRESHOLD", "0.3"))
# IOU  = float(os.getenv("IOU_NMS", "0.5"))
# MODEL_URL = os.getenv("MODEL_URL", "")

# _session = None
# _model_path = "/tmp/model.onnx"

# # ---- SAS helpers ----
# def _parse_conn():
#     conn = os.getenv("AzureWebJobsStorage", "")
#     parts = dict(p.split("=", 1) for p in conn.split(";") if "=" in p)
#     return parts.get("AccountName"), parts.get("AccountKey")

# def _make_blob_sas_url(container: str, name: str) -> str:
#     acct, key = _parse_conn()
#     if not acct or not key:
#         raise RuntimeError("AzureWebJobsStorage account/key not found.")
#     start  = datetime.utcnow() - timedelta(minutes=10)
#     expiry = datetime.utcnow() + timedelta(hours=6)
#     token = generate_blob_sas(
#         account_name=acct,
#         container_name=container,
#         blob_name=name,
#         account_key=key,
#         permission=BlobSasPermissions(read=True),
#         start=start,
#         expiry=expiry,
#     )
#     return f"https://{acct}.blob.core.windows.net/{container}/{name}?{token}"

# # ---- video opener ----
# def _open_video(url: str, name: str):
#     cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#     if cap is not None and cap.isOpened():
#         logging.info("video.open.ok", extra=cd(video=name, backend="opencv-ffmpeg"))
#         return cap, "opencv-ffmpeg"

#     logging.warning("video.open.try_imageio", extra=cd(video=name))
#     try:
#         import imageio.v3 as iio
#         reader = iio.imiter(url, plugin="FFMPEG")  # raw SAS
#         class _CapIter:
#             def __init__(self, it):
#                 self.it = it
#                 self.opened = True
#             def isOpened(self): return self.opened
#             def read(self):
#                 try:
#                     frame_rgb = next(self.it)
#                     bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#                     return True, bgr
#                 except StopIteration:
#                     self.opened = False
#                     return False, None
#             def release(self):
#                 self.opened = False
#         cap2 = _CapIter(reader)
#         logging.info("video.open.ok", extra=cd(video=name, backend="imageio-ffmpeg"))
#         return cap2, "imageio-ffmpeg"
#     except Exception as e:
#         logging.error("video.open.failed", extra=cd(video=name, err=str(e)))
#         return None, None

# # ---- model/session + inference helpers (unchanged from your working code) ----
# def _ensure_model():
#     global _session
#     if _session is not None:
#         return _session
#     if not os.path.exists(_model_path):
#         r = requests.get(MODEL_URL, timeout=60)
#         r.raise_for_status()
#         with open(_model_path, "wb") as f:
#             f.write(r.content)
#     so = ort.SessionOptions()
#     so.intra_op_num_threads = 2
#     _session = ort.InferenceSession(_model_path, sess_options=so, providers=["CPUExecutionProvider"])
#     return _session

# def _preprocess(img, size=640):
#     h, w = img.shape[:2]
#     scale = min(size / w, size / h)
#     nw, nh = int(w * scale), int(h * scale)
#     resized = cv2.resize(img, (nw, nh))
#     canvas = np.zeros((size, size, 3), dtype=np.uint8)
#     top = (size - nh) // 2; left = (size - nw) // 2
#     canvas[top:top + nh, left:left + nw] = resized
#     x = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
#     x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
#     return x, scale, left, top

# def _iou(b, others):
#     xx1 = np.maximum(b[0], others[:, 0]); yy1 = np.maximum(b[1], others[:, 1])
#     xx2 = np.minimum(b[2], others[:, 2]); yy2 = np.minimum(b[3], others[:, 3])
#     inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
#     area1 = (b[2] - b[0]) * (b[3] - b[1]); area2 = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
#     return inter / np.maximum(1e-6, area1 + area2 - inter)

# def _nms(boxes, scores, iou=0.5):
#     idxs = scores.argsort()[::-1]
#     keep = []
#     while idxs.size > 0:
#         i = idxs[0]
#         keep.append(i)
#         if idxs.size == 1: break
#         ious = _iou(boxes[i], boxes[idxs[1:]])
#         idxs = idxs[1:][ious < iou]
#     return keep

# def _postprocess(pred, scale, left, top, conf=0.3, iou=0.5):
#     p = pred[0]
#     if p.ndim == 3: p = p[0]
#     obj = p[:, 4:5]
#     cls = p[:, 5:]
#     cls_idx = np.argmax(cls, axis=1)
#     cls_conf = cls[np.arange(cls.shape[0]), cls_idx]
#     scores = (obj[:, 0] * cls_conf)
#     mask = scores >= conf
#     if not np.any(mask):
#         return []
#     p = p[mask]; scores = scores[mask]; cls_idx = cls_idx[mask]
#     xy = p[:, 0:2]; wh = p[:, 2:4]
#     x1y1 = xy - wh / 2; x2y2 = xy + wh / 2
#     boxes = np.concatenate([x1y1, x2y2], axis=1)
#     boxes[:, [0, 2]] -= left; boxes[:, [1, 3]] -= top
#     boxes /= scale
#     kept = _nms(boxes, scores, iou=iou)
#     return [(boxes[i].tolist(), float(scores[i]), int(cls_idx[i])) for i in kept]

# # ---- function entry ----
# def main(seg: dict):
#     sess = _ensure_model()

#     name   = seg["name"]
#     run_id = seg.get("runId", "")
#     start  = float(seg["start"])
#     end    = float(seg["end"])
#     fps_target = float(seg.get("fps", 1.0))

#     # Ignore incoming SAS; mint a fresh one to avoid 403
#     sas_url = _make_blob_sas_url("videos-in", name)

#     cap, backend = _open_video(sas_url, name)
#     if not cap or not cap.isOpened():
#         logging.error("segment.open_failed", extra=cd(runId=run_id, video=name))
#         return []

#     t0 = time.time()
#     frames_done = 0
#     next_emit = start
#     out = []

#     # Seek/time tracking per backend
#     if backend == "opencv-ffmpeg":
#         cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
#         def get_pos_s():
#             return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#     else:
#         pos = start
#         def get_pos_s():
#             return pos

#     while True:
#         cur = get_pos_s()
#         if cur >= end:
#             break

#         ok, frame = cap.read()
#         if not ok:
#             break

#         if backend != "opencv-ffmpeg":
#             pos += 1.0 / max(fps_target, 1e-6)

#         if cur < next_emit:
#             continue
#         next_emit = cur + (1.0 / max(fps_target, 1e-6))

#         x, scale, left, top = _preprocess(frame, 640)
#         pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]
#         dets = _postprocess(pred, scale, left, top, conf=CONF, iou=IOU)
#         frames_done += 1
#         for box, score, cls_id in dets:
#             out.append({
#                 "t": round(cur, 3),
#                 "cls": cls_id,
#                 "score": round(score, 3),
#                 "box": [round(v, 1) for v in box]
#             })

#     try: cap.release()
#     except Exception: pass

#     dt = time.time() - t0
#     logging.info("segment.done", extra=cd(
#         runId     = run_id,
#         video     = name,
#         start     = start,
#         seconds   = end - start,
#         frames    = frames_done,
#         duration_s= round(dt, 3)
#     ))
#     return out
