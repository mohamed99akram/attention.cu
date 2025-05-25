import torch
import numpy as np
import torch.nn.functional as F
import time

path = "data/Q.txt"
a1 = np.loadtxt(path, skiprows=1, delimiter=" ", dtype=np.float32)
Q = torch.tensor(a1)


path = "data/K.txt"
a2 = np.loadtxt(path, skiprows=1, delimiter=" ", dtype=np.float32)
K = torch.tensor(a2)


path = "data/V.txt"
a3 = np.loadtxt(path, skiprows=1, delimiter=" ", dtype=np.float32)
V = torch.tensor(a3)

times = []
if torch.cuda.is_available():
    # Multiple calls to warm up PyTorch's Cache
    for i in range(5):
        Qd, Kd, Vd = (t.cuda(non_blocking=True) for t in (Q, K, V))
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()

        scores = Qd @ Kd.T / (Qd.shape[-1] ** 0.5)
        attn   = F.softmax(scores, dim=-1)
        out_d  = attn @ Vd
        torch.cuda.synchronize()
        
        times.append(time.perf_counter() - t0)
        out_host = out_d.cpu()
    # Measure the time for the actual computation
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    print(f"PyTorch Average GPU time: {avg_time:.6f} seconds")
    print(f"PyTorch Max GPU time: {max_time:.6f} seconds")
    print(f"PyTorch Min GPU time: {min_time:.6f} seconds")

    np.savetxt("data/result_gpu_py.txt", out_host, fmt="%.8f", delimiter=" ", header=f"{out_host.shape[0]} {out_host.shape[1]}")
