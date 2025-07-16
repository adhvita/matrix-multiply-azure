import logging


def matrix_multiply(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    mid = n // 2

    def split(matrix):
        return [
            [row[:mid] for row in matrix[:mid]],
            [row[mid:] for row in matrix[:mid]],
            [row[:mid] for row in matrix[mid:]],
            [row[mid:] for row in matrix[mid:]]
        ]

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    def add(X, Y):
        return [[X[i][j] + Y[i][j] for j in range(len(X))] for i in range(len(X))]

    def subtract(X, Y):
        return [[X[i][j] - Y[i][j] for j in range(len(X))] for i in range(len(X))]

    M1 = matrix_multiply(add(A11, A22), add(B11, B22))
    logging.info(f"M1:, {M1}")
    M2 = matrix_multiply(add(A21, A22), B11)
    print("M2:", M2)
    M3 = matrix_multiply(A11, subtract(B12, B22))
    print("M3:", M3)
    M4 = matrix_multiply(A22, subtract(B21, B11))
    print("M4:", M4)
    M5 = matrix_multiply(add(A11, A12), B22)
    print("M5:", M5)
    M6 = matrix_multiply(subtract(A21, A11), add(B11, B12))
    print("M6:", M6)
    M7 = matrix_multiply(subtract(A12, A22), add(B21, B22))
    print("M7:", M7)

    C11 = add(subtract(add(M1, M4), M5), M7)
    print("C11:", C11)
    C12 = add(M3, M5)
    print("C12:", C12)
    C21 = add(M2, M4)
    print("C21:", C21)
    C22 = add(subtract(add(M1, M3), M2), M6)
    print("C22:", C22)

    new_matrix = []
    for i in range(mid):
        new_matrix.append(C11[i] + C12[i])
    for i in range(mid):
        new_matrix.append(C21[i] + C22[i])

    return new_matrix
