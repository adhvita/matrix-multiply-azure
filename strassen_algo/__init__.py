import logging
import json
import numpy as np
import azure.functions as func
from .strassen_module import strassen
from datetime import datetime, timezone


def main(inputblob: func.InputStream, outputblob: func.Out[str]):
    try:
        logging.info("Function triggered: Reading input blob '%s'", inputblob.name)
        # Step 1: Read the full synthetic_dataset.json content
        logging.info("Function triggered")
        content = inputblob.read().decode('utf-8')
        matrix_pairs = json.loads(content)

        results = []

        # Step 2: Loop through each A-B pair
        for idx, pair in enumerate(matrix_pairs):
            A = np.array(pair["A"])
            B = np.array(pair["B"])

            if A.shape != B.shape or A.shape[0] != A.shape[1]:
                logging.info("Processing matrix pair %d: shape %s", idx, A.shape)
                results.append({
                    "index": idx,
                    "error": "Matrix shapes are invalid or mismatched."
                })
                continue
            
            logging.info("Processing matrix pair %d: shape %s", idx, A.shape)
            C = strassen(A, B)
            results.append({
                "index": idx,
                "A": pair["A"],
                "B": pair["B"],
                "C": C.tolist()
            })

        # Step 3: Save results to output blob
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_pairs": len(results),
            "results": results
        }
        logging.info("✅ Processed %d matrix pairs from blob '%s'", len(results), inputblob.name)

        outputblob.set(json.dumps(output_data, indent=2))
        logging.info("Processed %d matrix pairs from %s",
                     len(results), inputblob.name)
    
    except Exception as e:
        logging.exception(" Error processing synthetic dataset")
        raise
