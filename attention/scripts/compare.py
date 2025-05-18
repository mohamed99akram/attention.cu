import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two matrices in two files')
    parser.add_argument('path1', type=str, help='Path to first matrix file')
    parser.add_argument('path2', type=str, help='Path to second matrix file')
    args = parser.parse_args()
    
    # Load matrices skipping the header line
    matrix1 = np.loadtxt(args.path1, skiprows=1, dtype=np.float32)

    matrix2 = np.loadtxt(args.path2, skiprows=1, dtype=np.float32)

    print(f"Matrix1 shape: {matrix1.shape}")
    print(f"Matrix2 shape: {matrix2.shape}")
    
    if matrix1.shape != matrix2.shape:
        print("ERROR: Matrices have different shapes and cannot be compared!")
        exit(1)

    # Absolute difference
    diff = np.abs(matrix1 - matrix2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")

    # Optional: check if all elements are close within a tolerance
    if np.allclose(matrix1, matrix2, atol=1e-6):
        print("Matrices are approximately equal within tolerance 1e-6")
    else:
        print("Matrices differ beyond tolerance 1e-6")
