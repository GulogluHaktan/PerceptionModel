"""Thermal soak on the Orin: run an engine flat out for several minutes in chunks and watch latency, temperature and power.
Usage: python scripts/orin_soak.py <engine.plan> [chunks=12] [chunk_s=25]
Prints one line per chunk: median ms, throughput, mean tj/GPU temperature (C), mean VDD_IN watts. A drop in throughput
together with a rising temperature means thermal throttling. The dev-kit fan is on; an enclosed drone may behave worse."""
import json, re, shutil, subprocess, sys, threading, time

plan = sys.argv[1]
chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 12
chunk_s = int(sys.argv[3]) if len(sys.argv) > 3 else 25
trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
rows, stop = [], threading.Event()


def sample():
    p = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        tj = re.search(r"tj@([\d.]+)C", line); gp = re.search(r"gpu@([\d.]+)C", line); w = re.search(r"VDD_IN (\d+)mW", line)
        if tj and gp and w:
            rows.append((time.time(), float(tj.group(1)), float(gp.group(1)), int(w.group(1))))
        if stop.is_set():
            break
    p.terminate()


th = threading.Thread(target=sample, daemon=True); th.start(); time.sleep(3)
out = []
for c in range(chunks):
    t0 = time.time()
    r = subprocess.run([trtexec, f"--loadEngine={plan}", "--warmUp=500", f"--duration={chunk_s}", "--avgRuns=100", "--noDataTransfers"],
                       capture_output=True, text=True)
    t1 = time.time()
    txt = r.stdout + r.stderr
    med = (re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", txt) or [None, None])[1]
    qps = (re.search(r"Throughput: ([\d.]+) qps", txt) or [None, None])[1]
    seg = [x for x in rows if t0 + 3 <= x[0] <= t1]
    d = {"chunk": c, "t_min": round((t1 - rows[0][0]) / 60, 1) if rows else None, "median_ms": med, "qps": qps,
         "tj_C": round(sum(x[1] for x in seg) / len(seg), 1) if seg else None, "gpu_C": round(sum(x[2] for x in seg) / len(seg), 1) if seg else None,
         "watts": round(sum(x[3] for x in seg) / len(seg) / 1000, 2) if seg else None}
    out.append(d); print(json.dumps(d), flush=True)
stop.set()
