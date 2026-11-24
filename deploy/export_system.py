"""Export the flown gate-perception system (DroneRacing "min14" configuration, handoff Updates 125-134) for the vehicle.

Two ONNX graphs instead of the research code's three passes of the main model per frame:
  system_448.onnx  image (1,3,448,448) RGB 0..1  ->  depth (1,1,448,448) | seg (1,11,448,448) | gates (1,K,9) main dense head, decoded |
                   cpk_heat (1,4,112,112) cpk_off (1,2,112,112) single-corner peak head | ng_gates (1,K,9) near-gate head, decoded |
                   pooled (1,128) mean fused stride-8 feature (condition router input)
                   = frozen core (FastViT-SA24 hybrid, re-parameterised) + CornerPeakHead + NearGateHead on ONE shared feature pass.
  refiner.onnx     patches (N,3,64,64) 0..1 (N dynamic) -> off (N,2), vis (N)   corner refiner (crops sampled on the host).
Also writes bundle.json (paths, sizes, post-processing constants) and reference outputs on test frames (ref_frames.npz) so the device can
check its TensorRT engines against this PyTorch reference (deploy/check_device.py).
Run (dev PC): env -u PYTHONPATH .venv/bin/python deploy/export_system.py [out_dir=deploy/bundle]"""
import glob, json, os, sys, shutil
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from pfm.model import PerceptionModel
from pfm.data import MEAN, STD
from pfm.cornerpeak import CornerPeakHead, NearGateHead
from pfm.refine import CornerRefiner, P as PATCH

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "deploy/bundle"); OUT.mkdir(parents=True, exist_ok=True)
S = int(os.environ.get("PFM_S", "448"))          # network input side (Update 165: 384 / 320 for the drone 60 FPS target)
DRONE = os.environ.get("PFM_DRONE", "") == "1"     # Update 165: system graph = core + gate/depth heads only; SegRefine v1 moves into the seg2 head engine (run every N-th frame)
CK = dict(core=Path(os.environ.get("PFM_CORE_CKPT", ROOT / "runs/release_v4/bigY_h_hybrid.pt")),      # PFM_CORE_CKPT + PFM_HYBRID_NAME: compatible student core (Update 165)
          cpk=Path(os.environ.get("PFM_CPK", Path.home() / "DroneRacing_eval/runs_cmp/cpk3.pt")), ng=Path(os.environ.get("PFM_NG", Path.home() / "DroneRacing_eval/runs_cmp/ng2.pt")),      # PFM_CPK / PFM_NG: 384 heads (Update 166)
          refiner=ROOT / "runs/refiner_v2/refiner.pt", seg=Path(os.environ.get("PFM_SEG_MODULE", ROOT / "runs/modules/seg_refine/module.pt")))
SEG_V2 = "seg_head2" in str(CK["seg"])          # SegHead2 (scripts/train_seg_module2.py) reads backbone stages_0 / stages_1 as well
SPLIT = os.environ.get("PFM_SPLIT", "") == "1"     # Update 160: system graph + separate head engines (seg2 = outdoor, indoor), chosen per frame by the scene router
INDOOR = os.environ.get("PFM_INDOOR", "")          # Update 159: indoor 22-class module (scripts/train_indoor_module.py) -> extra output "seg_in" (int32 class map)
SEG2_ENS = os.environ.get("PFM_SEG2_ENS", str(ROOT / "runs/modules/seg_head2/module.pt"))     # Update 157: seg = mean(v1 SegRefine, v2 SegHead2) logits; "" = v1 only
for k, p in CK.items(): assert p.exists(), p


class System(nn.Module):
    def __init__(self, core, cpk, ng, segm=None, segm2=None, inhead=None, taps=False):
        super().__init__(); self.core, self.cpk, self.ng, self.segm, self.segm2, self.inhead, self.taps = core, cpk, ng, segm, segm2, inhead, taps; self.cand = True     # cand: gate heads output top-M candidates, selection on the host (gdet_post.py)
        self.register_buffer("mean", MEAN.clone().view(1, 3, 1, 1)); self.register_buffer("std", STD.clone().view(1, 3, 1, 1))

    def forward(self, img01):
        x = (img01 - self.mean) / self.std; m = self.core
        net = m.backbone.net; h = net.stem_2(net.stem_1(net.stem_0(x))); s4 = net.stages_0(h); s8 = net.stages_1(s4); fmap = net.stages_2(s8)   # = m.backbone(x), taps kept
        p4, p3, p2 = m.pyramid(fmap)
        o = m.dense((p4, p3, p2), x.shape[-2:])
        f = p4 + F.interpolate(p3, size=p4.shape[-2:], mode="bilinear", align_corners=False) + F.interpolate(p2, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        heat, off = self.cpk(img01, f)
        if self.segm is None: seg = o["seg"]
        elif SEG_V2: seg = self.segm(img01, s4.float(), s8.float(), f.float(), o["seg"].float())      # plug-in seg module v2 (Update 156)
        else:
            seg = self.segm(img01, f, o["seg"].float())                                           # plug-in SegRefine module (Update 153)
            if self.segm2 is not None: seg = (seg + self.segm2(img01, s4.float(), s8.float(), f.float(), o["seg"].float())) / 2     # + SegHead2, mean of logits (Update 157)
        if self.cand: gates, ng = gdet_candidates(o["gdet"], GDET_K), gdet_candidates(self.ng(img01, f), GDET_K)
        else: gates, ng = o["slots"], m.dense.decode_gdet(self.ng(img01, f))
        out = (o["depth"][:, :1].float(), seg.float(), gates.float(), heat.float(), off.float(), ng.float(), f.float().mean((2, 3)))
        if self.inhead is not None: out = out + (self.inhead(img01, s4.float(), s8.float(), f.float(), o["seg"].float()).argmax(1).int(),)      # indoor class map
        if self.taps: out = out + (s4.float(), s8.float(), f.float(), o["seg"].float(), fmap.float())          # inputs of the separate head engines (split bundle)
        return out


class Seg2HeadWrap(nn.Module):
    """Outdoor head engine: seg = mean(SegRefine v1 logits (system output "seg"), SegHead2 v2 logits) = the monolithic v1+v2 bundle."""
    def __init__(self, segm2): super().__init__(); self.segm2 = segm2
    def forward(self, image, s4, s8, feat, seg_core, seg): return ((seg + self.segm2(image, s4, s8, feat, seg_core)) / 2).float()


class RepeatDWConv(nn.Module):
    """Exact rewrite of a grouped conv with groups == in_channels and out = m * in (FastViT downsample: 7x7 s2, m = 2): repeat every input
    channel m times, then a plain depthwise conv (groups == out_channels) with the SAME weight tensor (out, 1, k, k). TensorRT has no fast fp16
    kernel for the multiplier-2 grouped form and ran it as one fp32 GEMM per group (192 launches, 4.4 ms of the 20 ms engine on the Orin)."""
    def __init__(self, conv):
        super().__init__(); self.m = conv.out_channels // conv.in_channels
        self.conv = nn.Conv2d(conv.out_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, groups=conv.out_channels, bias=conv.bias is not None)
        with torch.no_grad():
            self.conv.weight.copy_(conv.weight)
            if conv.bias is not None: self.conv.bias.copy_(conv.bias)
    def forward(self, x):
        b, c, h, w = x.shape
        return self.conv(x[:, :, None].expand(b, c, self.m, h, w).reshape(b, c * self.m, h, w))


def fix_grouped_convs(module):
    """Replace every multiplier-m grouped conv (groups == in_channels < out_channels) by RepeatDWConv, in place. Returns the count."""
    n = 0
    for name, ch in list(module.named_children()):
        if isinstance(ch, nn.Conv2d) and ch.groups > 1 and ch.groups == ch.in_channels and ch.out_channels > ch.in_channels and ch.out_channels % ch.in_channels == 0:
            setattr(module, name, RepeatDWConv(ch)); n += 1
        else: n += fix_grouped_convs(ch)
    return n


class Seg12HeadWrap(nn.Module):
    """Drone bundle outdoor head: mean(SegRefine v1, SegHead2 v2) logits from the taps alone (v1 is no longer in the system graph)."""
    def __init__(self, segm, segm2): super().__init__(); self.segm, self.segm2 = segm, segm2
    def forward(self, image, s4, s8, feat, seg_core): return ((self.segm(image, feat, seg_core) + self.segm2(image, s4, s8, feat, seg_core)) / 2).float()


class IndoorHeadWrap(nn.Module):
    """Indoor head engine: 22-class indoor map (int32)."""
    def __init__(self, inhead): super().__init__(); self.inhead = inhead
    def forward(self, image, s4, s8, feat, seg_core, s16=None):
        return (self.inhead(image, s4, s8, feat, seg_core, s16) if s16 is not None else self.inhead(image, s4, s8, feat, seg_core)).argmax(1).int()


from gdet_post import gdet_candidates, gdet_select
GDET_K = int(os.environ["PFM_GDET_K"]); GDET = dict(k=GDET_K, dedup=0.25, main_levels=2, ng_levels=1, merge=True)


def select_both(gates_c, ng_c):
    """(1,M,11) candidates of the two gate heads -> (1,K,9) each, = decode_gdet."""
    return (gdet_select(gates_c[0], GDET_K, GDET["main_levels"], GDET["dedup"], GDET["merge"])[None], gdet_select(ng_c[0], GDET_K, GDET["ng_levels"], GDET["dedup"], GDET["merge"])[None])


def load_models():
    core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load(CK["core"], map_location="cpu")); core.eval()
    cpk = CornerPeakHead(); cpk.load_state_dict(torch.load(CK["cpk"], map_location="cpu")); cpk.eval()
    ng = NearGateHead(); ng.load_state_dict(torch.load(CK["ng"], map_location="cpu")); ng.eval()
    ref = CornerRefiner(); ref.load_state_dict(torch.load(CK["refiner"], map_location="cpu")); ref.eval()
    sys.path.insert(0, str(ROOT / "scripts"))
    if SEG_V2: from train_seg_module2 import SegHead2; segm = SegHead2()
    else: from train_seg_module import SegRefine; segm = SegRefine()
    segm.load_state_dict(torch.load(CK["seg"], map_location="cpu")); segm.eval()
    segm2 = None
    if SEG2_ENS and not SEG_V2:
        from train_seg_module2 import SegHead2; segm2 = SegHead2(); segm2.load_state_dict(torch.load(SEG2_ENS, map_location="cpu")); segm2.eval()
    inhead = None
    if INDOOR:
        from train_indoor_module import IndoorHead, IndoorHead2, IndoorHead3
        sd = torch.load(INDOOR, map_location="cpu")
        if "query.weight" in sd:                            # ARCH 7 mask-classification student (scripts/train_indoor_mask.py)
            from train_indoor_mask import IndoorMask; inhead = IndoorMask(core.backbone.net, w=int(sd["mask_feat.weight"].shape[1]), nq=int(sd["query.weight"].shape[0]))
            inhead.load_state_dict(sd); inhead.eval(); w = None
        else: w = int(sd["out.weight"].shape[1])                  # decoder width of the checkpoint (W env at training time)
        if w is None: pass
        elif any(k.startswith("stem.") for k in sd): from train_indoor_module import IndoorHead5; inhead = IndoorHead5(core.backbone.net, w=w)      # ARCH 5 (whole-backbone copy)
        elif any(k.startswith("stage1.") for k in sd): from train_indoor_module import IndoorHead4; inhead = IndoorHead4(core.backbone.net.stages_1, core.backbone.net.stages_2, w=w)
        elif any(k.startswith("stage2.") for k in sd): inhead = IndoorHead3(core.backbone.net.stages_2, w=w)
        elif any(k.startswith("ppm.") for k in sd): inhead = IndoorHead2(w=w)
        else: inhead = IndoorHead(w=w)
        if w is not None: inhead.load_state_dict(sd); inhead.eval()
    return core, cpk, ng, ref, segm, segm2, inhead


class RefinerWrap(nn.Module):
    def __init__(self, r): super().__init__(); self.r = r
    def forward(self, patches): off, vis, _ = self.r(patches); return off, vis


def test_frames(n=24):
    """Real gate frames (golden sets) + sim frames: what the device parity check runs on."""
    fs = sorted(glob.glob(str(ROOT / "data/real_mrg/golden/images/red_*.jpg")))[::7][:8] + sorted(glob.glob(str(ROOT / "data/real_tii/golden/images/*.jpg")))[::9][:8]
    fs += sorted(glob.glob(str(ROOT / "data/train_sim/images/**/*.png"), recursive=True))[::5003][:8]
    return [f for f in fs if os.path.exists(f)][:n]


def main():
    from timm.utils import reparameterize_model
    core, cpk, ng, ref, segm, segm2, inhead = load_models()
    if SPLIT and DRONE: heads = dict(seg2=Seg12HeadWrap(segm, segm2).eval(), indoor=inhead); segm = segm2 = inhead = None
    elif SPLIT: heads = dict(seg2=segm2, indoor=inhead); segm2 = inhead = None; assert heads["seg2"] is not None and heads["indoor"] is not None, "split bundle needs PFM_SEG2_ENS and PFM_INDOOR"
    sys_ref = System(core, cpk, ng, segm, segm2, inhead, taps=SPLIT).eval()                                  # reference: as trained
    sys_dep = System(reparameterize_model(core), cpk, ng, segm, segm2, inhead, taps=SPLIT).eval()           # deployed: multi-branch blocks merged
    if os.environ.get("PFM_FIX_GCONV", "1") == "1": print("grouped convs rewritten for TensorRT:", fix_grouped_convs(sys_dep), flush=True)
    fs = test_frames(); xs = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255 for f in fs]
    names = ["depth", "seg", "gates", "cpk_heat", "cpk_off", "ng_gates", "pooled"] + (["seg_in"] if inhead is not None else []) + (["s4", "s8", "feat", "seg_core", "s16"] if SPLIT else [])
    sp = OUT / f"system_{S}.onnx"
    torch.onnx.export(sys_dep, xs[0], str(sp), opset_version=17, input_names=["image"], output_names=names, dynamo=False)
    rp = OUT / "refiner.onnx"; pz = torch.rand(4, 3, PATCH, PATCH)
    torch.onnx.export(RefinerWrap(ref), pz, str(rp), opset_version=17, input_names=["patches"], output_names=["off", "vis"], dynamic_axes={"patches": {0: "n"}, "off": {0: "n"}, "vis": {0: "n"}}, dynamo=False)
    import onnx, onnxruntime as ort
    for p in (sp, rp): onnx.checker.check_model(str(p))
    ss = ort.InferenceSession(str(sp), providers=["CPUExecutionProvider"]); rs = ort.InferenceSession(str(rp), providers=["CPUExecutionProvider"])
    diff = {n: 0.0 for n in names}; diff["split_mismatch"] = 0; ref_out = {n: [] for n in names}; same_gates = 0
    with torch.no_grad():
        for x in xs:
            sys_ref.cand = False; t = [a.numpy() for a in sys_ref(x)]; sys_ref.cand = True; tc = [a.numpy() for a in sys_ref(x)]
            o = ss.run(None, {"image": x.numpy()}); o[2], o[5] = select_both(o[2], o[5]); s2, s5 = select_both(tc[2], tc[5])
            for a, b2 in ((t[2], s2), (t[5], s5)):
                va = a[0, :, 0] > -19.9; vb = b2[0, :, 0] > -19.9; split_ok = bool((va == vb).all()) and (not va.any() or float(np.abs(a[0, va] - b2[0, vb]).max()) < 1e-4)
                diff["split_mismatch"] = diff.get("split_mismatch", 0) + int(not split_ok)
            for n, a, b in zip(names, t, o):
                ref_out[n].append(a)
                if n not in ("gates", "ng_gates"): diff[n] = max(diff[n], float(np.abs(a - b).max()))
            ka, kb = t[2][0, :, 0] > np.log(0.05 / 0.95), o[2][0, :, 0] > np.log(0.05 / 0.95); same_gates += int((ka == kb).all())
            if (ka == kb).all() and ka.any(): diff["gates"] = max(diff["gates"], float(np.abs(t[2][0, ka, 1:] - o[2][0, ka, 1:]).max() * S))
        ref2 = CornerRefiner(); ref2.load_state_dict(torch.load(CK["refiner"], map_location="cpu")); ref2.eval()      # fresh copy: export tracing leaves the traced module unusable for the check
        rdiff = 0.0
        for n in (1, 5, 12):
            pz = torch.rand(n, 3, PATCH, PATCH); rt = [a.numpy() for a in RefinerWrap(ref2)(pz)]; ro = rs.run(None, {"patches": pz.numpy()}); rdiff = max([rdiff] + [float(np.abs(a - b).max()) for a, b in zip(rt, ro)])
    head_files, head_diff = {}, {}
    if SPLIT:                    # head engines + their reference outputs (torch heads on the torch taps) + ONNX check
        sp2 = OUT / "seg2_head.onnx"; spi = OUT / "indoor_head.onnx"; W2, WI = (heads["seg2"] if DRONE else Seg2HeadWrap(heads["seg2"]).eval()), IndoorHeadWrap(heads["indoor"]).eval()
        with torch.no_grad():
            v2in = hasattr(heads["indoor"], "ppm") and not hasattr(heads["indoor"], "stage2"); in_names = ["image", "s4", "s8", "feat", "seg_core"] + (["s16"] if v2in else [])
            WI_dep = IndoorHeadWrap(reparameterize_model(heads["indoor"])).eval() if hasattr(heads["indoor"], "stage2") else WI     # v3: its own backbone stage, re-parameterised for the engine
            t0 = sys_ref(xs[0]); a2 = (xs[0], t0[7], t0[8], t0[9], t0[10]) + (() if DRONE else (t0[1],)); ai = (xs[0], t0[7], t0[8], t0[9], t0[10]) + ((t0[11],) if v2in else ())
            torch.onnx.export(W2, a2, str(sp2), opset_version=17, input_names=["image", "s4", "s8", "feat", "seg_core"] + ([] if DRONE else ["seg"]), output_names=["seg_out"], dynamo=False)
            torch.onnx.export(WI_dep, ai, str(spi), opset_version=17, input_names=in_names, output_names=["seg_in"], dynamo=False)
            s2s = ort.InferenceSession(str(sp2), providers=["CPUExecutionProvider"]); sis = ort.InferenceSession(str(spi), providers=["CPUExecutionProvider"])
            ref_out["seg_outdoor"], ref_out["seg_in"] = [], []; head_diff = dict(seg2_logits=0.0, indoor_agree=1.0)
            for x in xs:
                t = sys_ref(x); a = (x, t[7], t[8], t[9], t[10]) + (() if DRONE else (t[1],)); ai = (x, t[7], t[8], t[9], t[10]) + ((t[11],) if v2in else ()); r2 = W2(*a).numpy(); ri = WI(*ai).numpy()
                ref_out["seg_outdoor"].append(r2.argmax(1).astype(np.uint8)); ref_out["seg_in"].append(ri.astype(np.uint8))
                have2 = {i.name for i in s2s.get_inputs()}; feed = {k: v.numpy() for k, v in zip(["image", "s4", "s8", "feat", "seg_core", "seg"], a) if k in have2}
                head_diff["seg2_logits"] = max(head_diff["seg2_logits"], float(np.abs(s2s.run(None, feed)[0] - r2).max()))
                have = {i.name for i in sis.get_inputs()}; feed_i = {k: v.numpy() for k, v in zip(in_names, ai) if k in have}      # unused inputs are dropped by the exporter
                head_diff["indoor_agree"] = min(head_diff["indoor_agree"], float((sis.run(None, feed_i)[0] == ri).mean()))
        head_files = dict(seg2=sp2.name, indoor=spi.name)
        for k in ("s4", "s8", "feat", "seg_core", "s16"): ref_out.pop(k)
    R = {n: np.concatenate(v) for n, v in ref_out.items()}; R["seg"] = R["seg"].argmax(1).astype(np.uint8); R["depth"] = R["depth"].astype(np.float16)   # compact: class map, fp16 depth
    if "seg_in" in R: R["seg_in"] = R["seg_in"].astype(np.uint8)
    np.savez_compressed(OUT / "ref_frames.npz", images=(np.concatenate([x.numpy() for x in xs]) * 255).round().astype(np.uint8), **R, files=np.array([os.path.relpath(f, ROOT) for f in fs]))
    bundle = dict(
        version="release_v4 + cpk3 + ng2 + refiner_v2 (min14) + seg modules (SegRefine v1 + SegHead2 v2, mean of logits)" + (" | drone: system = core + gate/depth heads, seg modules in the seg2 head" if DRONE else ""), input_size=S, drone=DRONE, input="RGB float 0..1, NCHW, squashed (not letterboxed) to 448x448; ImageNet normalisation is inside the graph",
        graphs=dict(system=sp.name, refiner=rp.name), outputs=names, gates_format="candidates (1,M,11+M): score, level, logit, 8 coords, same-gate row -> gdet_post.gdet_select", checkpoints={k: os.path.relpath(p, Path.home()) for k, p in CK.items()},
        seg_classes=["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"],
        seg_in_classes=(__import__("pfm.indoor_classes", fromlist=["IN_CLASSES"]).IN_CLASSES if inhead is not None else None),
        gdet=GDET, post=dict(thr=0.05, ng_thr=0.6, ng_min=0.0, cpk_thr=0.3, confirm=True, confirm_hi=0.5, confirm_r=0.1, confirm_rmin=6.0, confirm_n=2, snap=False,
                  dedup_all=True, dedup_iou=0.2, dedup_inside=False, min_size_px=14.0, refiner_margin=2.0, refiner_vis_thr=0.0),
        heads=head_files or None, head_onnx_check=head_diff or None,
        router=(json.loads((ROOT / "runs/modules/scene_router.json").read_text()) if SPLIT else None), router_hold=3,
        onnx_check=dict(frames=len(xs), max_abs_diff={k: round(v, 5) for k, v in diff.items()}, frames_same_gate_set=same_gates, refiner_max_abs_diff=round(rdiff, 6),
                        note="reference = PyTorch model as trained (not re-parameterised); gates diff in px at 448"),
        size_mb={p.name: round(p.stat().st_size / 2**20, 1) for p in [sp, rp] + [OUT / f for f in head_files.values()]})
    (OUT / "bundle.json").write_text(json.dumps(bundle, indent=1)); print(json.dumps(bundle["onnx_check"]), bundle["size_mb"])


if __name__ == "__main__":
    main()
