
# python3 scripts/matmul.py

# python scripts/matmul.py


import numpy as np

a1 = np.loadtxt("data/matrix1.txt", skiprows=1, delimiter=" ", dtype=np.float32)

a2 = np.loadtxt("data/matrix2.txt", skiprows=1, delimiter=" ", dtype=np.float32)

result = np.matmul(a1, a2.T)
np.savetxt("data/result_py.txt", result, fmt="%.6f", delimiter=" ", header=f"{result.shape[0]} {result.shape[1]}")
