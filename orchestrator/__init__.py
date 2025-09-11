import azure.durable_functions as df
try:
    from shared.logging_utils import jlog  # uses your enriched jlog (profile/phase/op/host_arch)
except Exception:
    jlog = None  
def orchestrator_function(context: df.DurableOrchestrationContext):
    cfg = context.get_input()
    # CHANGE: derive a deterministic run_id (never generate UUIDs in an orchestrator)
    # Prefer the router-provided run_id; otherwise derive from input blob name (deterministic).
    run_id = cfg.get("run_id") or (cfg.get("input_blob", "run_unknown").rsplit(".", 1)[0])

    # Optional: surface progress (safe/deterministic)
    # CHANGE: set custom status so you can watch stages in the portal
    context.set_custom_status({"stage": "prepare_tiles", "run_id": run_id})
    if jlog and not context.is_replaying:
        jlog({
            "phase": "e2e",
            "op": "orchestrate_start",
            "run_id": run_id,
            "tile_size": tile,
            "dtype": dtype,
            "strassen_threshold": thr,
            "success": True
        })
    # 1) Pre-split NPZ into tiles (A_i_q.npy, B_q_j.npy)
    manifest = yield context.call_activity("activity_worker", {
        "op": "prepare_tiles",
        "run_id": run_id, 
        "input_container": cfg["input_container"],
        "input_blob": cfg["input_blob"],
        "temp_container": cfg["temp_container"],
        "tile": int(cfg["tile_size"]),
        "dtype": cfg["dtype"]
    })
    N = int(manifest["N"])
    tile = int(manifest["tile"])
    tiles = int(manifest["tiles"])
    thr = int(cfg["strassen_threshold"])
    # 2) Fan-out multiplies (row/col across q) → partials
    # CHANGE: schedule as a PARALLEL wave using task_all (true fan-out)
    context.set_custom_status({"stage": "multiply", "run_id": run_id})
    
    multiply_tasks = []
    for i in range(tiles):
        for j in range(tiles):
            multiply_tasks.append(context.call_activity("activity_worker", {
                "op": "multiply_tile_rowcol",
                "run_id": run_id,                          # CHANGE: propagate run_id
                "temp_container": cfg["temp_container"],
                "i": i, "j": j,
                "tiles": tiles,
                "tile": tile,
                "N": N,                                    # CHANGE: include N for logging records
                "dtype": cfg["dtype"],
                "strassen_threshold": thr
            }))
    partial_refs = yield context.task_all(multiply_tasks)  # CHANGE: parallel fan-out complete

    # # 2) Fan-out multiplies (row/col across q) → partials
    # partial_refs = []
    # for i in range(tiles):
    #     for j in range(tiles):
    #         part = yield context.call_activity("activity_worker", {
    #             "op": "multiply_tile_rowcol",
    #             "temp_container": cfg["temp_container"],
    #             "i": i, "j": j,
    #             "tiles": tiles,
    #             "tile": tile,
    #             "dtype": cfg["dtype"],
    #             "strassen_threshold": thr
    #         })
    #         partial_refs.append(part)

    # # 3) Reductions → final tiles C_i_j.npy
    # final_tiles = []
    # for part in partial_refs:
    #     fin = yield context.call_activity("activity_worker", {
    #         "op": "reduce_partials",
    #         "temp_container": cfg["temp_container"],
    #         "partials": part["partials"],
    #         "i": part["i"], "j": part["j"],
    #         "tile": tile, "dtype": cfg["dtype"]
    #     })
    #     final_tiles.append(fin)

    # 3) Reductions → final tiles C_i_j.npy
    # CHANGE: schedule reductions as another PARALLEL wave
    context.set_custom_status({"stage": "reduce", "run_id": run_id})
    reduce_tasks = []
    for part in partial_refs:
        reduce_tasks.append(context.call_activity("activity_worker", {
            "op": "reduce_partials",
            "run_id": run_id,                              # CHANGE: propagate run_id
            "temp_container": cfg["temp_container"],
            "partials": part["partials"],
            "i": part["i"], "j": part["j"],
            "tile": tile,
            "N": N,                                        # CHANGE: include N for logging records
            "dtype": cfg["dtype"]
        }))
    final_tiles = yield context.task_all(reduce_tasks)     # CHANGE: parallel fan-in complete
    # 4) Merge tiles → single C.npy in output container
    context.set_custom_status({"stage": "merge", "run_id": run_id})
    out_blob = yield context.call_activity("activity_worker", {
        "op": "merge_tiles",
        "run_id": run_id,                                  # CHANGE: propagate run_id
        "temp_container": cfg["temp_container"],
        "output_container": cfg["output_container"],
        "N": N, "tile": tile, "tiles": tiles, "dtype": cfg["dtype"]
    })
    # CHANGE: final status for easy discovery in the portal / queries
    context.set_custom_status({
        "stage": "done",
        "run_id": run_id,
        "output": f"{cfg['output_container']}/{out_blob}",
        "N": N, "tile": tile, "tiles": tiles
    })
    if jlog and not context.is_replaying:
        jlog({
            "phase": "e2e",
            "op": "merge",
            "run_id": run_id,
            "count": len(final_tiles),
            "success": True
        })
        jlog({
            "phase": "e2e",
            "op": "orchestrate_end",
            "run_id": run_id,
            "success": True
        })
    # CHANGE: return extra metadata (tiles/tile) to help your results scripts
    return {
        "output": f"{cfg['output_container']}/{out_blob}",
        "N": N,
        "dtype": cfg["dtype"],
        "tile": tile,
        "tiles": tiles,
        "run_id": run_id
    }

    # # 4) Merge tiles → single C.npy in output container
    # out_blob = yield context.call_activity("activity_worker", {
    #     "op": "merge_tiles",
    #     "temp_container": cfg["temp_container"],
    #     "output_container": cfg["output_container"],
    #     "N": N, "tile": tile, "tiles": tiles, "dtype": cfg["dtype"]
    # })

    # return { "output": f"{cfg['output_container']}/{out_blob}", "N": N, "dtype": cfg["dtype"] }

main = df.Orchestrator.create(orchestrator_function)
