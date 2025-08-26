import numpy as np

def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def pad_to_square_pow2(A: np.ndarray, B: np.ndarray, logger):
    n = A.shape[0]
    assert A.shape == B.shape and A.ndim == 2 and n == A.shape[1], "A,B must be square and same size"
    P = next_pow2(n)
    if P == n:
        logger.debug("No padding needed (power-of-two)")
        return A, B, n
    logger.info(f"Padding from {n} to {P} (power-of-two)")
    Ap = np.zeros((P, P), dtype=A.dtype)
    Bp = np.zeros((P, P), dtype=B.dtype)
    Ap[:n, :n] = A
    Bp[:n, :n] = B
    return Ap, Bp, n

def combine_quadrants(C11, C12, C21, C22):
    n2 = C11.shape[0]
    C = np.empty((n2*2, n2*2), dtype=C11.dtype)
    C[:n2, :n2] = C11;  C[:n2, n2:] = C12
    C[n2:, :n2] = C21;  C[n2:, n2:] = C22
    return C

def strassen_square(A: np.ndarray, B: np.ndarray, threshold: int, logger, depth: int=0) -> np.ndarray:
    n = A.shape[0]
    if n <= threshold:
        logger.debug(f"[depth={depth}] base n={n}")
        return A.dot(B)

    n2 = n // 2
    A11 = A[:n2, :n2]; A12 = A[:n2, n2:]
    A21 = A[n2:, :n2]; A22 = A[n2:, n2:]
    B11 = B[:n2, :n2]; B12 = B[:n2, n2:]
    B21 = B[n2:, :n2]; B22 = B[n2:, n2:]

    logger.debug(f"[depth={depth}] split n={n} -> {n2}")

    M1 = strassen_square(A11 + A22, B11 + B22, threshold, logger, depth+1)
    M2 = strassen_square(A21 + A22, B11,         threshold, logger, depth+1)
    M3 = strassen_square(A11,         B12 - B22, threshold, logger, depth+1)
    M4 = strassen_square(A22,         B21 - B11, threshold, logger, depth+1)
    M5 = strassen_square(A11 + A12,   B22,       threshold, logger, depth+1)
    M6 = strassen_square(A21 - A11,   B11 + B12, threshold, logger, depth+1)
    M7 = strassen_square(A12 - A22,   B21 + B22, threshold, logger, depth+1)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    return combine_quadrants(C11, C12, C21, C22)

def strassen_rectangular(A: np.ndarray, B: np.ndarray, threshold: int, logger) -> np.ndarray:
    Ap, Bp, n = pad_to_square_pow2(A, B, logger)
    Cpad = strassen_square(Ap, Bp, threshold, logger)
    return Cpad[:n, :n].copy()
