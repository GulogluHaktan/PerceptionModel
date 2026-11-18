"""Catalogue growth, end to end (one GPU model at a time): frame -> VLM finds important objects that are not in the catalogue -> SAM2 instance crop -> Hunyuan (image_to_asset.py) -> automatic gate -> registered with provenance.
  .venv/bin/python scripts/grow_catalogue.py <frame.png> [--known tag,tag] [--max_objects 2] [--vlm qwen3-vl:8b-instruct] [--licence "own render"] [--dry]
The model never gives sizes or code: only a tag from TAG_SIZE_M and a box. Output: assets/generated/<name>.{glb,usd,json,_crop.png}; json has status accepted|rejected|duplicate, the gate result and provenance. Only 'accepted' assets are offered to the resolver (scripts/layout_to_job_gen.py).
Needs: Ollama running (qwen3-vl), the Hunyuan env in /home/haktan/Projects/i23d. Do NOT run while Isaac renders."""
import argparse, base64, datetime, gc, hashlib, io, json, os, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); os.chdir(Path(__file__).resolve().parents[1])
from pfm import asset_growth as G                                    # noqa: E402
from pfm.assets import REGISTRY                                      # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("frame"); ap.add_argument("--known", default=""); ap.add_argument("--max_objects", type=int, default=2); ap.add_argument("--vlm", default="qwen3-vl:8b-instruct")
ap.add_argument("--licence", default="unspecified"); ap.add_argument("--dry", action="store_true", help="detect + crop only"); ap.add_argument("--no_consistency", action="store_true", help="skip the second-seed check (saves ~1 min per object)"); ap.add_argument("--out", default="assets/generated"); ap.add_argument("--host", default="http://localhost:11434")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
frame = Image.open(a.frame).convert("RGB"); W, H = frame.size; known = [t for t in a.known.split(",") if t]
meta_all = [json.load(open(p)) for p in sorted(out.glob("*.json"))]

# 1) VLM detections (constrained to the tag enum), then unload the model so the GPU is free for SAM2/Hunyuan
buf = io.BytesIO(); frame.save(buf, "PNG")
body = {"model": a.vlm, "stream": False, "think": False, "keep_alive": 0, "format": G.detection_schema(), "options": {"temperature": 0.2, "num_ctx": 8192},
        "messages": [{"role": "user", "content": G.detect_prompt(known), "images": [base64.b64encode(buf.getvalue()).decode()]}]}
raw = json.loads(urllib.request.urlopen(urllib.request.Request(a.host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())["message"]["content"]
dets = G.parse_detections(raw, known)[: a.max_objects]; print("VLM raw:", raw[:400]); print("kept detections:", json.dumps(dets))
if not dets:
    sys.exit("no new objects found")

# 2) SAM2 instance crops (box prompt)
from transformers import Sam2Model, Sam2Processor                    # noqa: E402
dev = "cuda"; sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny"); sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(dev).eval()
crops = []
for i, d in enumerate(dets):
    x0, y0, x1, y1 = d["box"]; box = [x0 * W, y0 * H, x1 * W, y1 * H]
    with torch.no_grad():
        sin = sp(images=frame, input_boxes=[[box]], return_tensors="pt").to(dev); so = sm(**sin, multimask_output=False)
    mask = sp.post_process_masks(so.pred_masks.cpu(), sin["original_sizes"])[0][0, 0].numpy() > 0
    ys, xs = np.where(mask)
    if len(ys) < 0.005 * W * H:
        print("object", i, "mask too small, skipped"); continue
    pad = int(0.06 * max(ys.max() - ys.min(), xs.max() - xs.min())); yy0, yy1, xx0, xx1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    rgba = np.dstack([np.asarray(frame), (mask * 255).astype(np.uint8)])[yy0:yy1, xx0:xx1]; crops.append((d, Image.fromarray(rgba), mask[yy0:yy1, xx0:xx1]))
del sm, sp; gc.collect(); torch.cuda.empty_cache()

# 3-5) per object: duplicate check, generation (separate process, own env), gate, register
for d, crop, mask in crops:
    h = G.dhash(crop.convert("RGB")); dup = G.is_duplicate(h, d["tag"], meta_all)
    name = f"gen_{d['tag']}_{hashlib.sha1(np.asarray(crop).tobytes()).hexdigest()[:6]}"; cp = out / f"{name}_crop.png"
    if dup:
        print(name, "duplicate of " + dup, "- skipped"); continue
    crop.save(cp)
    if a.dry:
        print(name, "dry run: crop saved", cp); continue
    env = dict(os.environ, PYTHONPATH=os.environ.get("I23D", "/home/haktan/Projects/i23d") + "/shim:" + os.environ.get("I23D", "/home/haktan/Projects/i23d") + "/triposr")
    subprocess.run([os.environ.get("I23D", "/home/haktan/Projects/i23d") + "/.venv/bin/python", "scripts/image_to_asset.py", str(cp), d["tag"], name, "--out", str(out)], check=True, env=env, stdout=subprocess.DEVNULL)
    import trimesh
    alt = None
    if not a.no_consistency:                                         # second generation with another seed: low agreement = the photo does not decide the shape
        alt_dir = out / "_alt"; subprocess.run([os.environ.get("I23D", "/home/haktan/Projects/i23d") + "/.venv/bin/python", "scripts/image_to_asset.py", str(cp), d["tag"], name, "--seed", "1", "--out", str(alt_dir)], check=True, env=env, stdout=subprocess.DEVNULL)
        am = trimesh.load(alt_dir / f"{name}.glb", force="mesh"); alt = (am.vertices, am.faces); import shutil; shutil.rmtree(alt_dir)
    m = trimesh.load(out / f"{name}.glb", force="mesh"); D = json.load(open("assets/models/dims.json")); prior = G.proportion_prior(d["tag"], D, REGISTRY)
    g = G.gate(m.vertices, m.faces, mask, d["tag"], prior, alt=alt); meta = json.load(open(out / f"{name}.json"))
    meta.update({"status": "accepted" if g["ok"] else "rejected", "gate": g, "prior_used": prior is not None, "crop_dhash": f"{h:x}", "crop": str(cp),
                 "provenance": {"source_frame": a.frame, "source_licence": a.licence, "vlm": a.vlm, "detection": d, "date": datetime.date.today().isoformat(), "shape_model": meta["generator"]}})
    json.dump(meta, open(out / f"{name}.json", "w"), indent=1); print(name, meta["status"], json.dumps(g["metrics"]), g["reasons"])
