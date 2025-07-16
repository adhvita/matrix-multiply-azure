def validate_matrices(matrix_a, matrix_b):
    if not matrix_a or not matrix_b:
        raise ValueError("Input matrices cannot be empty.")
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Matrix A's column count must match Matrix B's row count.")