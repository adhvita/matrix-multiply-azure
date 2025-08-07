# import logging
# import json
# import numpy as np
# import azure.functions as func
# from .strassen_module import strassen
# from datetime import datetime, timezone


# def main(inputblob: func.InputStream, outputblob: func.Out[str]):
#     try:
#         logging.info("Function triggered: Reading input blob '%s'", inputblob.name)
#         # Step 1: Read the full synthetic_dataset.json content
#         logging.info("Function triggered")
#         content = inputblob.read().decode('utf-8')
#         matrix_pairs = json.loads(content)

#         results = []

#         # Step 2: Loop through each A-B pair
#         for idx, pair in enumerate(matrix_pairs):
#             A = np.array(pair["A"])
#             B = np.array(pair["B"])

#             if A.shape != B.shape or A.shape[0] != A.shape[1]:
#                 logging.info("Processing matrix pair %d: shape %s", idx, A.shape)
#                 results.append({
#                     "index": idx,
#                     "error": "Matrix shapes are invalid or mismatched."
#                 })
#                 continue
            
#             logging.info("Processing matrix pair %d: shape %s", idx, A.shape)
#             C = strassen(A, B)
#             results.append({
#                 "index": idx,
#                 "A": pair["A"],
#                 "B": pair["B"],
#                 "C": C.tolist()
#             })

#         # Step 3: Save results to output blob
#         output_data = {
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#             "num_pairs": len(results),
#             "results": results
#         }
#         logging.info("✅ Processed %d matrix pairs from blob '%s'", len(results), inputblob.name)

#         outputblob.set(json.dumps(output_data, indent=2))
#         logging.info("Processed %d matrix pairs from %s",
#                      len(results), inputblob.name)
    
#     except Exception as e:
#         logging.exception(" Error processing synthetic dataset")
#         raise

import logging
import json
import numpy as np
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from .strassen_module import strassen
from datetime import datetime, timezone
import os


def main(event: func.EventGridEvent):
    try:
        logging.info("🔔 EventGrid trigger fired")

        event_data = event.get_json()
        blob_url = event_data['url']
        logging.info(f"📥 Blob URL: {blob_url}")

        # Extract blob info
        account_url = f"https://{os.environ['AZURE_STORAGE_ACCOUNT_NAME']}.blob.core.windows.net"
        container_name = blob_url.split("/")[3]
        blob_name = "/".join(blob_url.split("/")[4:])

        # Connect to Blob service
        blob_service_client = BlobServiceClient(account_url=account_url, credential=os.environ['AZURE_STORAGE_KEY'])
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Read blob content
        blob_data = blob_client.download_blob().readall()
        matrix_pairs = json.loads(blob_data.decode('utf-8'))

        results = []

        # Process each matrix pair
        for idx, pair in enumerate(matrix_pairs):
            A = np.array(pair["A"])
            B = np.array(pair["B"])

            if A.shape != B.shape or A.shape[0] != A.shape[1]:
                results.append({
                    "index": idx,
                    "error": "Matrix shapes are invalid or mismatched."
                })
                continue

            logging.info(f"🔄 Processing matrix pair {idx} with shape {A.shape}")
            C = strassen(A, B)
            results.append({
                "index": idx,
                "A": pair["A"],
                "B": pair["B"],
                "C": C.tolist()
            })

        # Prepare output content
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_pairs": len(results),
            "results": results
        }

        # Write to output container
        output_blob_name = f"result-{os.path.basename(blob_name)}"
        output_blob_client = blob_service_client.get_blob_client(container="output-container", blob=output_blob_name)
        output_blob_client.upload_blob(json.dumps(output_data, indent=2), overwrite=True)
        logging.info(f"✅ Output written to output-container/{output_blob_name}")

    except Exception as e:
        logging.exception("❌ Error processing blob event")
        raise
