import azure.functions as func
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_lib'))
from shared_lib.strassen_unit_intermediates import intermediate_3_handler

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        result = intermediate_3_handler(req_body)
        return func.HttpResponse(json.dumps({"result": result}), status_code=200)
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)