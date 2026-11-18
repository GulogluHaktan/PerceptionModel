"""Export a trained student to ONNX (fixed input size, batch 1) and check it against PyTorch.
Outputs: depth (1,1,S,S) relative depth, seg (1,11,S,S) class logits, gates (1,4,9) = [score logit, 4 corners (u,v) image-normalised], decoded in the graph
(peak NMS + duplicate removal, same as DenseHeads.decode_gdet), gdet (1,9,G,G) raw dense gate map. Input: image (1,3,S,S), RGB in 0..1, ImageNet mean/std applied INSIDE the graph.
The ViT position embedding is resampled once to the export size (timm does the same resampling on the fly, which ONNX cannot express); FastViT is re-parameterised.
HALF=1 writes model_<size>_fp16.onnx (needs the GPU for tracing; the check then compares fp16 ONNX on CPU against fp32 PyTorch).
Usage: [HALF=1] PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/bigD KIND=vit python scripts/export_onnx.py [size=448] [check images glob]
Writes <CKPT>/<KIND>/model_<size>.onnx and prints the largest output difference ONNX Runtime (CPU) vs PyTorch on real images. TensorRT: see scripts/trt_build.sh."""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import MEAN, STD
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); S = int(sys.argv[1]) if len(sys.argv) > 1 else 448; IMGS = sys.argv[2] if len(sys.argv) > 2 else "data/real_mrg/golden/images/red_*.jpg"; HALF = os.environ.get("HALF") == "1"; dev = "cuda" if HALF else "cpu"
sd = torch.load(f"{CK}/{K}/final.pt", map_location="cpu"); m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(sd); m.eval()
ref = m       # reference = the model as trained (dynamic position embedding)
if K == "vit":
    import copy, timm
    from timm.layers import resample_abs_pos_embed
    e = copy.deepcopy(m); old = e.backbone.net
    net = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, img_size=S, global_pool="").eval()
    st = old.state_dict(); st["pos_embed"] = resample_abs_pos_embed(st["pos_embed"], (S // 16, S // 16), num_prefix_tokens=0 if old.no_embed_class else old.num_prefix_tokens)
    net.load_state_dict(st); e.backbone.net = net
else:
    from timm.utils import reparameterize_model
    e = reparameterize_model(m)
class Wrap(torch.nn.Module):
    def __init__(self, m, half=False): super().__init__(); self.m = m; self.half_ = half; self.register_buffer("mean", MEAN.clone()); self.register_buffer("std", STD.clone())
    def forward(self, x):
        x = (x - self.mean) / self.std
        o = self.m(x.half() if self.half_ else x, with_teacher_heads=False)       # HALF: weights and activations fp16, input/outputs stay fp32
        return tuple(t.float() for t in (o["depth"], o["seg"], o["slots"]) + ((o["gdet"],) if "gdet" in o else ()))
w = Wrap(e, HALF).eval()
if HALF: w.m.half(); w.to(dev)      # fp16 graph for TensorRT 11 (strongly typed networks: precision comes from the ONNX file, there is no fp16 builder flag any more)
names = ["depth", "seg", "gates"] + (["gdet"] if e.dense.gdet is not None else []); out = f"{CK}/{K}/model_{S}{'_fp16' if HALF else ''}.onnx"
files = sorted(glob.glob(IMGS))[:: max(1, len(glob.glob(IMGS)) // 12)][:12]
xs = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255 for f in files]
torch.onnx.export(w, xs[0].to(dev), out, opset_version=17, input_names=["image"], output_names=names, dynamo=False)
import onnx, onnxruntime as ort
onnx.checker.check_model(out); sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"]); diff = {n: 0.0 for n in names}; gate_px = 0.0; same_n = 0
with torch.no_grad():
    for x in xs:
        t = Wrap(ref)(x); o = sess.run(None, {"image": x.numpy()})
        for n, a, b in zip(names, t, o):
            if n != "gates": diff[n] = max(diff[n], float(np.abs(a.numpy() - b).max()))
        ta, ob = t[2][0].numpy(), o[2][0]; ka, kb = ta[:, 0] > 0, ob[:, 0] > 0; same_n += int((ka == kb).all())
        if (ka == kb).all() and ka.any(): gate_px = max(gate_px, float(np.abs((ta[ka, 1:] - ob[ka, 1:]) * np.tile([256, 192], 4)).max()))
res = {"onnx": out, "size_mb": round(os.path.getsize(out) / 2**20, 1), "check_images": len(xs), "max_abs_diff": {k: round(v, 5) for k, v in diff.items() if k != "gates"}, "frames_with_same_gate_set": same_n, "max_gate_corner_diff_px": round(gate_px, 4)}
print(json.dumps(res)); json.dump(res, open(out.replace(".onnx", "_onnx_check.json"), "w"), indent=1)
