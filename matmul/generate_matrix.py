import numpy as np
np.random.seed(0)  # For reproducibility

def generate_matrix(rows, cols):
    return np.random.rand(rows, cols)

def save_matrix(matrix, filename, rows, cols):
    # save to file : row col \n matrix
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        for row in matrix:
            # row as .6f
            f.write(" ".join(map(str, [f"{x:.6f}" for x in row])) + "\n")

if __name__ == "__main__":
    # Example usage
    rows = 5
    cols = 4
    filename = "matrix1.txt"
    matrix = generate_matrix(rows, cols)
    save_matrix(matrix, filename, rows, cols)