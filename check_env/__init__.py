import json, sys
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        import numpy as np
        payload = {
            "ok": True,
            "python": sys.version,
            "numpy_version": np.__version__,
            "numpy_path": getattr(np, "__file__", None),
        }
        return func.HttpResponse(json.dumps(payload), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"ok": False, "error": repr(e)}),
            status_code=500,
            mimetype="application/json",
        )
