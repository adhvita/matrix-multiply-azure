# matrix-multiply-azure/shared_lib/generate_dataset.py
import numpy as np
import csv

def generate_matrix_pair(height, width):
    matrix_a = np.random.randint(0, 100, (height, width)).tolist()
    matrix_b = np.random.randint(0, 100, (height, width)).tolist()
    return matrix_a, matrix_b

def save_matrix_pairs_to_csv(file_path, experiments):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['matrix_a', 'matrix_b'])
        for exp in experiments:
            height = exp['matA']['height']
            width = exp['matA']['width']
            for _ in range(10):  # Example: 10 entries per experiment configuration
                matrix_a, matrix_b = generate_matrix_pair(height, width)
                writer.writerow([matrix_a, matrix_b])

if __name__ == "__main__":
    experiments = [
        {"matA": {"height": 600, "width": 600}},
        {"matA": {"height": 800, "width": 800}},
        {"matA": {"height": 400, "width": 400}},
        {"matA": {"height": 200, "width": 200}}
    ]
    save_matrix_pairs_to_csv("matrix_dataset.csv", experiments)
