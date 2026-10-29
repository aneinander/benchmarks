"""mem-bw.py - measured GPU memory bandwidth with torch, printed as "GBPS MIB".

Copies a buffer of random floats 20 times and counts read + write. Random
data on purpose: Arc compresses uniform buffers, so a zero-filled test reads
far too high (1227 GB/s with zeros vs 533 GB/s real on a B70). Best of three
rounds. Tries 1 GiB, then smaller buffers when the inference server has taken
most of the VRAM. Prints nothing when there is no usable GPU.

Runs with any python that has torch for the GPU: collect-env.sh runs it inside
the server's own container (docker exec), or with $BW_PYTHON.
"""
import sys
import time

import torch

if hasattr(torch, "xpu") and torch.xpu.is_available():
    dev, sync = "xpu", torch.xpu.synchronize
elif torch.cuda.is_available():          # NVIDIA, and AMD through ROCm
    dev, sync = "cuda", torch.cuda.synchronize
else:
    sys.exit(0)

for mib in (1024, 512, 256):
    try:
        x = torch.rand(mib * 2**20 // 4, device=dev)
        y = torch.empty_like(x)
        break
    except RuntimeError:                 # out of memory
        continue
else:
    sys.exit(0)

y.copy_(x)
sync()
best = 0.0
for _ in range(3):
    t = time.perf_counter()
    for _ in range(20):
        y.copy_(x)
    sync()
    best = max(best, 20 * 2 * x.numel() * 4 / (time.perf_counter() - t) / 1e9)
print(f"{best:.0f} {mib}")
