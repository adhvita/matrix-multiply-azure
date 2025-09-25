import os, cv2, logging, time
from datetime import datetime, timedelta
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

def cd(**k):  # custom dimensions helper
    return {'custom_dimensions': k}

# ---- SAS helpers ----
def _parse_conn():
    """Parse AzureWebJobsStorage connection string for account name/key."""
    conn = os.getenv("AzureWebJobsStorage", "")
    parts = dict(p.split("=", 1) for p in conn.split(";") if "=" in p)
    return parts.get("AccountName"), parts.get("AccountKey")

def _make_blob_sas_url(container: str, name: str) -> str:
    """Fresh read SAS with 10 min negative skew to avoid clock issues."""
    acct, key = _parse_conn()
    if not acct or not key:
        raise RuntimeError("AzureWebJobsStorage account/key not found.")
    start  = datetime.utcnow() - timedelta(minutes=10)
    expiry = datetime.utcnow() + timedelta(hours=6)
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

# ---- video opener (FFmpeg preferred, imageio fallback) ----
def _open_video(url: str, name: str):
    # Prefer OpenCV+FFMPEG to avoid CAP_IMAGES % pattern parsing
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap is not None and cap.isOpened():
        logging.info("video.open.ok", extra=cd(video=name, backend="opencv-ffmpeg"))
        return cap, "opencv-ffmpeg"

    logging.warning("video.open.try_imageio", extra=cd(video=name))
    try:
        import imageio.v3 as iio

        reader = iio.imiter(url, plugin="FFMPEG")  # pass raw SAS
        class _CapIter:
            def __init__(self, it):
                self.it = it
                self.opened = True
            def isOpened(self): return self.opened
            def read(self):
                try:
                    frame_rgb = next(self.it)          # HxWx3 RGB uint8
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
def main(input: dict) -> float:
    run_id = input["runId"]
    name   = input["name"]
    sas    = input["sas"]

    logging.info("probe_video.begin", extra=cd(runId=run_id, video=name))

    # Try OpenCV first
    cap, backend = _open_video(sas, name)

    # If OpenCV said "opened" but can't read a first frame, fall back to imageio.
    if cap and cap.isOpened():
        ok, _ = cap.read()
        if not ok and backend == "opencv-ffmpeg":
            try:
                cap.release()
            except Exception:
                pass
            logging.warning("probe.switch_to_imageio", extra=cd(video=name))
            cap, backend = _open_video(sas, name)

    if not cap or not cap.isOpened():
        logging.error("probe.open_failed", extra=cd(runId=run_id, video=name))
        return 0.0

    try:
        duration_s = 0.0
        fps = 25.0

        if backend == "opencv-ffmpeg":
            # Try metadata first
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if frames > 0 and fps > 0:
                duration_s = float(frames / fps)
            else:
                # Fallback: sample frames quickly to estimate duration
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    duration_s += 1.0 / fps
        else:
            # imageio backend: try metadata; fallback to scan
            try:
                import imageio.v3 as iio
                meta = iio.immeta(sas, plugin="FFMPEG")
                fps = float(meta.get("fps", 25.0) or 25.0)
                duration_s = float(meta.get("duration", 0.0) or 0.0)
                if duration_s <= 0.0:
                    while True:
                        ok, _ = cap.read()
                        if not ok:
                            break
                        duration_s += 1.0 / fps
            except Exception:
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    duration_s += 1.0 / 25.0

        logging.info("probe_video.end", extra=cd(runId=run_id, video=name,
                                                duration_s=round(duration_s, 3), fps=fps))
        return float(duration_s)
    finally:
        try:
            cap.release()
        except Exception:
            pass

# ---- function entry ----
# def main(input: dict) -> float:
#     run_id = input["runId"]
#     name   = input["name"]
#     # NOTE: ignore incoming SAS; we mint our own to avoid 403
#     logging.info("probe_video.begin", extra=cd(runId=run_id, video=name))

#     sas_url = _make_blob_sas_url("videos-in", name)
#     cap, backend = _open_video(sas_url, name)
#     if not cap or not cap.isOpened():
#         logging.error("probe.open_failed", extra=cd(runId=run_id, video=name))
#         return 0.0

#     try:
#         fps = 25.0
#         duration_s = 0.0
#         if backend == "opencv-ffmpeg":
#             fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
#             frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
#             if frames > 0 and fps > 0:
#                 duration_s = float(frames / fps)
#             else:
#                 while True:
#                     ok, _ = cap.read()
#                     if not ok: break
#                     duration_s += 1.0 / fps
#         else:
#             # imageio fallback: try metadata, else scan
#             try:
#                 import imageio.v3 as iio
#                 meta = iio.immeta(sas_url, plugin="FFMPEG")
#                 fps = float(meta.get("fps", 25.0) or 25.0)
#                 duration_s = float(meta.get("duration", 0.0) or 0.0)
#                 if duration_s <= 0.0:
#                     while True:
#                         ok, _ = cap.read()
#                         if not ok: break
#                         duration_s += 1.0 / fps
#             except Exception:
#                 while True:
#                     ok, _ = cap.read()
#                     if not ok: break
#                     duration_s += 1.0 / fps

#         logging.info("probe_video.end", extra=cd(runId=run_id, video=name,
#                                                 duration_s=round(duration_s, 3), fps=fps))
#         return float(duration_s)
#     finally:
#         try: cap.release()
#         except Exception: pass