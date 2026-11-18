"""Measurement helpers (D9 draft): latency, peak memory, optional energy.

Rules (AgriJetsonBench-style, to confirm): warm-up, fixed power mode, report median+p95,
batch 1, fp16, include pre/post-processing separately. On Orin read power from tegrastats;
here we only do latency + memory.
"""
from __future__ import annotations

import statistics
import time

import torch


@torch.no_grad()
def latency(model, shape=(1, 3, 384, 384), n=50, warmup=10, device="cuda", half=True):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = model.eval().to(device)
    x = torch.randn(*shape, device=device)
    if half:
        model, x = model.half(), x.half()
    for _ in range(warmup):
        model(x, with_teacher_heads=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model(x, with_teacher_heads=False)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"median_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * len(ts)) - 1],
            "fps": 1000 / statistics.median(ts), "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
