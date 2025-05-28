import torch
import numpy as np
import torch.nn.functional as F

path = "data/Q.txt"
a1 = np.loadtxt(path, skiprows=1, delimiter=" ", dtype=np.float32)
Q = torch.tensor(a1).to('cuda' if torch.cuda.is_available() else 'cpu')


path = "data/K.txt"
a2 = np.loadtxt(path, skiprows=1, delimiter=" ", dtype=np.float32)
K = torch.tensor(a2).to('cuda' if torch.cuda.is_available() else 'cpu')


path = "data/V.txt"
a3 = np.loadtxt(path, skiprows=1, delimiter=" ", dtype=np.float32)
V = torch.tensor(a3).to('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Compute attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # [B, L, L]

# Step 2: Apply softmax
attn_weights = F.softmax(scores, dim=-1)  # [B, L, L]

# Step 3: Multiply by   
output = torch.matmul(attn_weights, V)  # [B, L, d_v]
output = output.cpu().numpy()  # Move to CPU and convert to numpy array
np.savetxt("data/result_py.txt", output, fmt="%.8f", delimiter=" ", header=f"{output.shape[0]} {output.shape[1]}")
