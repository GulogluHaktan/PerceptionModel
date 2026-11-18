"""Orin TensorRT benchmark: ONNX export -> trtexec (fp16) -> latency + power/energy (tegrastats).
Usage (on the Orin): python scripts/orin_trt_bench.py <vit|hybrid> [size=384] [hybrid_name=fastvit_sa24] [reparam]
Needs: onnx export deps (onnx), /usr/src/tensorrt/bin/trtexec or trtexec on PATH, tegrastats.
Reports GPU compute time (median/mean), throughput, mean watts (VDD_IN) and joules per frame during the timed run.
Power mode is NOT changed here: report which `nvpmodel -q` mode was active (decision 9: fixed power mode)."""
import json, re, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import torch
from pfm.energy import EnergyMeter
from pfm.model import PerceptionModel


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        o = self.m(x, with_teacher_heads=False)
        return o["depth"], o["seg"], o["corners"]


def export(student, size, hybrid_name, out, reparam=False):
    m = PerceptionModel(student, hybrid_name=hybrid_name).eval()
    if student == "vit":
        # timm's dynamic pos-embed resize uses antialiased bicubic, which ONNX cannot export:
        # rebuild the ViT with a fixed input size (random weights are fine for latency).
        import timm
        m.backbone.net = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,
                                           img_size=size, global_pool="").eval()
    if reparam:   # FastViT-style multi-branch blocks merged into single convs for inference
        from timm.utils import reparameterize_model
        m = reparameterize_model(m)
    x = torch.randn(1, 3, size, size)
    torch.onnx.export(Wrap(m), x, str(out), opset_version=17, input_names=["image"],
                      output_names=["depth", "seg", "corners"], dynamo=False)


def main():
    student = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    hname = sys.argv[3] if len(sys.argv) > 3 else "fastvit_sa24"
    reparam = len(sys.argv) > 4 and sys.argv[4] == "reparam"
    tag = f"{student}_{size}" + ("_reparam" if reparam else "")
    out = Path("runs/trt"); out.mkdir(parents=True, exist_ok=True)
    onnx = out / f"{tag}.onnx"
    export(student, size, hname, onnx, reparam)
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    mode = subprocess.run("nvpmodel -q 2>/dev/null | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    plan = out / f"{tag}.plan"
    if not plan.exists():   # engine build is slow and CPU-heavy: keep it OUT of the energy measurement
        subprocess.run([trtexec, f"--onnx={onnx}", "--fp16", f"--saveEngine={plan}"], capture_output=True, text=True, check=True)
    with EnergyMeter() as idle:                  # idle floor (same power mode, GPU quiet)
        time.sleep(6)
    cmd = [trtexec, f"--loadEngine={plan}", "--warmUp=2000", "--duration=20", "--avgRuns=100", "--noDataTransfers"]
    with EnergyMeter() as e:
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t
    txt = r.stdout + r.stderr
    (out / f"{tag}.trtexec.log").write_text(txt)
    g = lambda pat: (re.search(pat, txt) or [None, None])[1]
    res = {"student": student, "size": size, "reparam": reparam, "power_mode": mode, "exit": r.returncode,
           "gpu_median_ms": g(r"GPU Compute Time:.*?median = ([\d.]+) ms"), "gpu_mean_ms": g(r"GPU Compute Time:.*?mean = ([\d.]+) ms"),
           "throughput_qps": g(r"Throughput: ([\d.]+) qps"), "idle_watts": round(idle.mean_watts, 2),
           "active_watts": round(e.mean_watts, 2), "samples": len(e.samples), "run_s": round(wall, 1)}
    qps = float(res["throughput_qps"] or 0)
    if qps:
        res["joules_per_frame_total"] = round(e.mean_watts / qps, 3)
        res["joules_per_frame_net"] = round((e.mean_watts - idle.mean_watts) / qps, 3)   # minus idle floor
    print(json.dumps(res))


main()
