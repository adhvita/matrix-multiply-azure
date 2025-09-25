import azure.durable_functions as df
import os
import logging, uuid, time
def cd(**k):  # custom dimensions helper
    return {'custom_dimensions': k}

SEG_SECS  = 10.0     
FRAME_FPS = 1.0

def orchestrator_function(context: df.DurableOrchestrationContext):
    info = context.get_input()  # {name, sas, runId}
    logging.info("orchestrator.begin", extra=cd(runId=info["runId"], video=info["name"]))

    duration = yield context.call_activity("probe_video", info)  # float seconds

    # Build segments (ensure at least one)
    if not duration or duration <= 0:
        segments = [{
            "runId": info["runId"], "name": info["name"], "sas": info["sas"],
            "start": 0.0, "end": float(SEG_SECS), "seconds": float(SEG_SECS), "fps": FRAME_FPS
        }]
    else:
        segments = []
        t = 0.0
        while t < duration:
            end = min(t + SEG_SECS, duration)
            segments.append({
                "runId": info["runId"], "name": info["name"], "sas": info["sas"],
                "start": float(t), "end": float(end), "seconds": float(end - t), "fps": FRAME_FPS
            })
            t = end

    logging.info("orchestrator.segments",
                 extra=cd(runId=info["runId"], video=info["name"], count=len(segments),
                          duration_s=float(duration or 0.0)))

    tasks   = [context.call_activity("process_segment", s) for s in segments]
    results = yield context.task_all(tasks)
    flat    = [ev for sub in results for ev in sub]
    
    yield context.call_activity("save_results", {"runId": info["runId"], "name": info["name"], "events": flat})
    logging.info("orchestrator.end",
                 extra=cd(runId=info["runId"], video=info["name"], objects=len(flat)))
    return {"processed": info["name"], "objects": len(flat)}

# FRAME_FPS = int(os.getenv("FRAME_FPS", "1"))
# SEG_SECS  = int(os.getenv("SEGMENT_SECONDS", "60"))

# def orchestrator_function(context: df.DurableOrchestrationContext):
#     info = context.get_input()  # {name, sas}
#     run_id = info["runId"]; 
#     name = info["name"]; 
#     sas = info["sas"]
#     logging.info("orchestrator.begin", extra=cd(runId=info["runId"], video=info["name"]))

#     meta = yield context.call_activity("probe_video", {"runId": run_id, "name": name, "sas": sas})
#     duration = float(meta.get("duration_s", 0.0))

#     # guard: zero/unknown duration -> process a single segment window (first SEG_SECS)
#     if not duration or duration <= 0:
#         segments = [{
#         "runId": run_id, "name": name, "sas": sas,
#         "start": 0.0, "end": float(SEG_SECS),
#         "seconds": float(SEG_SECS),
#         "fps": FRAME_FPS
#     }]
#     else:
#         segments = []
#         t = 0.0
#         while t < duration:
#             end = min(t + SEG_SECS, duration)
#             segments.append({
#             "runId": run_id, "name": name, "sas": sas,
#             "start": t, "end": end,
#             "seconds": end - t,
#             "fps": FRAME_FPS
#         })
#             t = end

#     tasks = [context.call_activity("process_segment", s) for s in segments]
#     results = yield context.task_all(tasks)
#     flat = [item for sub in results for item in sub]

#     yield context.call_activity("save_results", {"runId": run_id, "name": name, "events": flat})
#     logging.info("orchestrator.end", extra=cd(runId=run_id, video=name, segments=len(segments), events=len(flat)))
#     return {"runId": run_id, "name": name, "segments": len(segments), "events": len(flat)}

main = df.Orchestrator.create(orchestrator_function)
