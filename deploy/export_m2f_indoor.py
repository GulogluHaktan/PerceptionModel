"""Export a COCO-panoptic Mask2Former (default Swin-Tiny) as the indoor 22-class module: image (1,3,H,W) RGB 0..1 -> int32 class map (1,H/4,W/4)... upsampled to H,W.
ImageNet normalisation, semantic inference (sum_q softmax(class) * sigmoid(mask)) and the 133 -> 22 mapping are inside the graph.
Run: env -u PYTHONPATH .venv/bin/python deploy/export_m2f_indoor.py [out_dir] [H W]   (env MODEL)"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); from pfm.indoor_classes import K_IN, coco_lut, IN_CLASSES
OUTDOOR = os.environ.get("CLASSSET") == "out"           # outdoor 19-class student (Cityscapes ids) -> output "seg_fine"
if OUTDOOR: from pfm.outdoor_classes import OUT_CLASSES as IN_CLASSES, K_OUT as K_IN
from transformers import Mask2FormerForUniversalSegmentation
MODEL = os.environ.get("MODEL", "facebook/mask2former-swin-tiny-coco-panoptic"); OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "deploy/m2f_swint"); OUT.mkdir(parents=True, exist_ok=True)
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (448, 448)


PLUGIN = os.environ.get("PLUGIN", "1") == "1"      # deformable attention -> TensorRT MultiscaleDeformableAttnPlugin_TRT (~75 % of the Orin time without it)
import transformers.models.mask2former.modeling_mask2former as _m2f
_orig_msda = _m2f.multi_scale_deformable_attention


class _MSDAPlugin(torch.autograd.Function):
    """Traced as ONE ONNX node 'MultiscaleDeformableAttnPlugin_TRT' (inputs: value (N,S,M,D), spatial_shapes (L,2) int32, level_start_index (L) int32,
    sampling_locations (N,Q,M,L,P,2), attention_weights (N,Q,M,L,P); output (N,Q,M,D)). Forward = the HF reference so traced shapes are right."""
    @staticmethod
    def forward(ctx, value, shapes, starts, loc, attn):
        hw = [(int(h), int(w)) for h, w in shapes.tolist()]
        return _orig_msda(value, hw, loc, attn).view(value.shape[0], loc.shape[1], value.shape[2], value.shape[3])
    @staticmethod
    def symbolic(g, value, shapes, starts, loc, attn):
        return g.op("MultiscaleDeformableAttnPlugin_TRT", value, shapes, starts, loc, attn)


def _msda_plugin(value, value_spatial_shapes, sampling_locations, attention_weights):
    hw = [(int(h), int(w)) for h, w in value_spatial_shapes]
    shapes = torch.tensor(hw, dtype=torch.int32); starts = torch.tensor([0] + list(np.cumsum([h * w for h, w in hw])[:-1]), dtype=torch.int32)
    out = _MSDAPlugin.apply(value.float(), shapes, starts, sampling_locations.float(), attention_weights.float())
    return out.flatten(2)


if PLUGIN: _m2f.multi_scale_deformable_attention = _msda_plugin


FT_CKPT = os.environ.get("FT_CKPT", "")            # our fine-tuned / distilled student (scripts/finetune_m2f_indoor.py: grouped 22+1 class head)


class M2FIndoor(nn.Module):
    def __init__(self):
        super().__init__()
        if FT_CKPT:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts")); import finetune_m2f_indoor as FT
            self.m = FT.build(); self.m.load_state_dict(torch.load(FT_CKPT, map_location="cpu")); self.m.eval()
            gp = self.m.class_predictor; self.m.class_predictor = gp.lin           # the grouped logsumexp head overflows in TensorRT fp16 (exp of logits > 11):
            A = torch.zeros(K_IN, gp.lin.out_features - 1)                          # use the 133+1 logits and sum member probabilities after the softmax instead (same values)
            for c, mem in enumerate(gp.groups): A[c, mem] = 1.0
        else:
            self.m = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).eval()
            lut = torch.tensor(coco_lut(self.m.config.id2label)); A = torch.zeros(K_IN, len(lut)); A[lut, torch.arange(len(lut))] = 1.0      # 22 x 133
        self.register_buffer("A", A)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, image):
        o = self.m(pixel_values=(image - self.mean) / self.std)
        cls = o.class_queries_logits.softmax(-1)[..., :-1] @ self.A.t()                    # (1,Q,22): sum of member probs
        seg = torch.einsum("bqc,bqhw->bchw", cls, o.masks_queries_logits.sigmoid())
        return F.interpolate(seg, size=image.shape[-2:], mode="bilinear", align_corners=False).argmax(1).int()


if __name__ == "__main__":
    net = M2FIndoor(); x = torch.rand(1, 3, H, W)
    with torch.no_grad(): ref = net(x)
    p = OUT / f"m2f_{'outdoor' if OUTDOOR else 'indoor'}_{H}x{W}{'_ft' if FT_CKPT else ''}{'_plugin' if PLUGIN else ''}.onnx"
    torch.onnx.export(net, x, str(p), opset_version=17, input_names=["image"], output_names=["seg_fine" if OUTDOOR else "seg_in"], dynamo=False)
    agree = None
    if not PLUGIN:                 # the plugin node only runs in TensorRT
        import onnxruntime as ort
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]); o = s.run(None, {"image": x.numpy()})[0]; agree = float((o == ref.numpy()).mean())
    print(json.dumps(dict(onnx=p.name, size_mb=round(p.stat().st_size / 2**20, 1), plugin=PLUGIN, onnx_vs_torch_agree=agree)))
    (OUT / "meta.json").write_text(json.dumps(dict(model=MODEL, input=[H, W], classes=IN_CLASSES, onnx_vs_torch_agree=agree)))
