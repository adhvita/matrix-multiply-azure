def lambda_benchmark_handler(event):
    return {"message": "Lambda benchmark executed", "event_keys": list(event.keys())}