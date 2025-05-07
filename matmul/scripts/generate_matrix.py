
# TO RUN: python3 scripts/generate_matrix.py 50 100  data/matrix2.txt

import numpy as np
import argparse
import os
def generate_matrix(rows, cols):
    return np.random.rand(rows, cols) * 1000

def save_matrix(matrix, filename, rows, cols):
    # save to file : row col \n matrix
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        for row in matrix:
            # row as .6f
            f.write(" ".join(map(str, [f"{x:.6f}" for x in row])) + "\n")

if __name__ == "__main__":
    # Example usage
    # rows = 4
    # cols = 5
    # filename = "matrix2.txt"
    parser = argparse.ArgumentParser(description='Generate a random matrix and save it to a file.')
    parser.add_argument('rows', type=int, help='Number of rows in the matrix')
    parser.add_argument('cols', type=int, help='Number of columns in the matrix')
    parser.add_argument('filename', type=str, help='Filename to save the matrix')
    args = parser.parse_args()
    
    
    matrix = generate_matrix(args.rows, args.cols)
    save_matrix(matrix, args.filename, args.rows, args.cols)