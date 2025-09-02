import azure.functions as func
import azure.durable_functions as df
import os, json

async def main(req: func.HttpRequest, starter: str) -> func.HttpResponse:
    client = df.DurableOrchestrationClient(starter)
    data = req.get_json()
    instance_id = await client.start_new("orchestrator", None, {
        "input_container": data.get("input_container","inputs"),
        "input_blob": data["input_blob"],     # e.g. "pair_7168x7168_2025...npz"
        "temp_container": os.getenv("TEMP_CONTAINER","temp"),
        "output_container": os.getenv("OUTPUT_CONTAINER","output-container"),
        "tile_size": int(os.getenv("TILE_SIZE","1024")),
        "dtype": os.getenv("MM_DTYPE","float32"),
        "strassen_threshold": int(os.getenv("STRASSEN_THRESHOLD","1024"))
    })
    return func.HttpResponse(json.dumps({"instanceId": instance_id}),
                             mimetype="application/json")
