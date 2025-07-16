def split_matrix(matrix, block_size):
    n = len(matrix)
    blocks = []
    for i in range(0, n, block_size):
        row_blocks = []
        for j in range(0, n, block_size):
            block = [row[j:j+block_size] for row in matrix[i:i+block_size]]
            row_blocks.append(block)
        blocks.append(row_blocks)
    return blocks

def merge_blocks(blocks):
    block_size = len(blocks[0][0])
    rows = len(blocks)
    cols = len(blocks[0])
    merged_matrix = []
    for row_block in blocks:
        for i in range(block_size):
            merged_row = []
            for block in row_block:
                merged_row.extend(block[i])
            merged_matrix.append(merged_row)
    return merged_matrix

if __name__ == "__main__":
    # Example usage for testing
    matrix = [[i + j * 4 for i in range(4)] for j in range(4)]
    block_size = 2
    blocks = split_matrix(matrix, block_size)
    merged = merge_blocks(blocks)
    print("Original Matrix:", matrix)
    print("Split Blocks:", blocks)
    print("Merged Matrix:", merged)