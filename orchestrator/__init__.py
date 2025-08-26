import azure.durable_functions as df

def orchestrator_function(context: df.DurableOrchestrationContext):
    cfg = context.get_input()

    # 1) Pre-split NPZ into tiles (A_i_q.npy, B_q_j.npy)
    manifest = yield context.call_activity("activity_worker", {
        "op": "prepare_tiles",
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
    partial_refs = []
    for i in range(tiles):
        for j in range(tiles):
            part = yield context.call_activity("activity_worker", {
                "op": "multiply_tile_rowcol",
                "temp_container": cfg["temp_container"],
                "i": i, "j": j,
                "tiles": tiles,
                "tile": tile,
                "dtype": cfg["dtype"],
                "strassen_threshold": thr
            })
            partial_refs.append(part)

    # 3) Reductions → final tiles C_i_j.npy
    final_tiles = []
    for part in partial_refs:
        fin = yield context.call_activity("activity_worker", {
            "op": "reduce_partials",
            "temp_container": cfg["temp_container"],
            "partials": part["partials"],
            "i": part["i"], "j": part["j"],
            "tile": tile, "dtype": cfg["dtype"]
        })
        final_tiles.append(fin)

    # 4) Merge tiles → single C.npy in output container
    out_blob = yield context.call_activity("activity_worker", {
        "op": "merge_tiles",
        "temp_container": cfg["temp_container"],
        "output_container": cfg["output_container"],
        "N": N, "tile": tile, "tiles": tiles, "dtype": cfg["dtype"]
    })

    return { "output": f"{cfg['output_container']}/{out_blob}", "N": N, "dtype": cfg["dtype"] }

main = df.Orchestrator.create(orchestrator_function)
