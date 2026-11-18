"""Shadow-mode test (decision 8): run several TensorRT engines AT THE SAME TIME on the Orin and measure
memory and latency under contention. Usage (on the Orin):
  python scripts/orin_shadow_bench.py runs/trt/hybrid_384_reparam.plan runs/trt/hybrid_384_reparam.plan [more.plan ...]
Reports per-engine median latency/throughput, unified RAM (and swap) before vs during, GPU load, mean watts.
Engines are built by scripts/orin_trt_bench.py. Duration 20 s, desktop should be idle."""
import json, re, shutil, subprocess, sys, threading, time

trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
plans = sys.argv[1:]
rows = []


def sample(stop):
    p = subprocess.Popen(["tegrastats", "--interval", "250"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        m = re.search(r"RAM (\d+)/(\d+)MB.*?SWAP (\d+)/(\d+)MB", line)
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        w = re.search(r"VDD_IN (\d+)mW", line)
        if m and g and w:
            rows.append((time.time(), int(m.group(1)), int(m.group(3)), int(g.group(1)), int(w.group(1))))
        if stop.is_set():
            break
    p.terminate()


stop = threading.Event()
th = threading.Thread(target=sample, args=(stop,), daemon=True); th.start()
time.sleep(4)
base_end = time.time()
procs = [subprocess.Popen([trtexec, f"--loadEngine={p}", "--warmUp=2000", "--duration=20", "--avgRuns=100", "--noDataTransfers"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for p in plans]
outs = [p.communicate()[0] for p in procs]
run_end = time.time()
stop.set(); th.join(timeout=2)
base = [r for r in rows if r[0] <= base_end]
run = [r for r in rows if base_end + 6 <= r[0] <= run_end - 1]   # skip start-up and tear-down
mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
res = {"engines": []}
for p, o in zip(plans, outs):
    g = lambda pat: (re.search(pat, o) or [None, None])[1]
    res["engines"].append({"plan": p, "exit_ok": "PASSED" in o or "Throughput" in o,
                           "median_ms": g(r"GPU Compute Time:.*?median = ([\d.]+) ms"), "qps": g(r"Throughput: ([\d.]+) qps")})
res.update({"ram_mb_before": mean([r[1] for r in base]), "ram_mb_during_mean": mean([r[1] for r in run]),
            "ram_mb_during_max": max([r[1] for r in run], default=None), "swap_mb_during_max": max([r[2] for r in run], default=None),
            "gpu_load_pct_mean": mean([r[3] for r in run]), "watts_mean": round(mean([r[4] for r in run]) / 1000, 2) if run else None,
            "power_mode": subprocess.run("nvpmodel -q | head -1", shell=True, capture_output=True, text=True).stdout.strip()})
print(json.dumps(res))
