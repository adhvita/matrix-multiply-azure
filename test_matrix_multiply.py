import requests
import numpy as np
import json

# Configuration
API_URL = "http://localhost:7071/api/MultiUnitMultiplication"  # Adjust as needed
matrix_size = 2  # You can increase this as needed (e.g., 100, 500)

# Generate random matrices
matrix_a = np.random.randint(0, 10, (matrix_size, matrix_size)).tolist()
matrix_b = np.random.randint(0, 10, (matrix_size, matrix_size)).tolist()

print("Matrix A:", matrix_a)
print("Matrix B:", matrix_b)

# Prepare payload
payload = {
    "matrix_a": matrix_a,
    "matrix_b": matrix_b
}

# Send request
response = requests.post(API_URL, json=payload)

# Show result
if response.status_code == 200:
    print("Result Matrix:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Request failed with status {response.status_code}: {response.text}")
