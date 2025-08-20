import numpy as np

def add(A, B):
    return A + B

def subtract(A, B):
    return A - B

def split(matrix):
    mid = matrix.shape[0] // 2
    return matrix[:mid, :mid], matrix[:mid, mid:], matrix[mid:, :mid], matrix[mid:, mid:]

def join(C11, C12, C21, C22):
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

def pad_matrix(M, size):
    padded = np.zeros((size, size), dtype=M.dtype)
    padded[:M.shape[0], :M.shape[1]] = M
    return padded

def strassen(A, B):
    assert A.shape == B.shape, "Matrices must be same shape for Strassen"
    n = A.shape[0]

    # Pad matrices if not power of 2
    m = next_power_of_2(n)
    if n != m:
        A = pad_matrix(A, m)
        B = pad_matrix(B, m)

    result = _strassen_recursive(A, B)

    # Remove padding
    return result[:n, :n]

def _strassen_recursive(A, B):
    n = A.shape[0]

    if n == 1:
        return A * B

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    M1 = _strassen_recursive(add(A11, A22), add(B11, B22))
    M2 = _strassen_recursive(add(A21, A22), B11)
    M3 = _strassen_recursive(A11, subtract(B12, B22))
    M4 = _strassen_recursive(A22, subtract(B21, B11))
    M5 = _strassen_recursive(add(A11, A12), B22)
    M6 = _strassen_recursive(subtract(A21, A11), add(B11, B12))
    M7 = _strassen_recursive(subtract(A12, A22), add(B21, B22))

    C11 = add(subtract(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(subtract(add(M1, M3), M2), M6)

    return join(C11, C12, C21, C22)