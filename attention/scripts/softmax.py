import torch
import numpy as np
import torch.nn.functional as F

path = "data/matrix1.txt"
a1 = np.loadtxt(path, skiprows=1, delimiter=" ", dtype=np.float32)
a1_t = torch.tensor(a1)

a1_softmax = F.softmax(a1_t, dim=1).cpu().numpy()
np.savetxt("data/result_py.txt", a1_softmax, fmt="%.8f", delimiter=" ", header=f"{a1_softmax.shape[0]} {a1_softmax.shape[1]}")
