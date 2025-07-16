import azure.functions as func
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_lib'))
from shared_lib.strassen import matrix_multiply
from shared_lib.utils import validate_matrices

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        matrix_a = req_body.get('matrix_a')
        matrix_b = req_body.get('matrix_b')
        validate_matrices(matrix_a, matrix_b)
        result = matrix_multiply(matrix_a, matrix_b)
        return func.HttpResponse(json.dumps({"result": result}), status_code=200)
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)