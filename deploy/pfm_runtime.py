"""On-vehicle gate perception runtime: camera frame -> gates (4 corners), single-corner peaks, segmentation, depth.

Same detections as the flown research configuration (DroneRacing "min14": ~/DroneRacing_eval/ours_detector.py with
OURS_THR=0.05, near-gate head thr 0.6, corner refiner, corner-peak confirmation, global dedup IoU 0.2, min gate size 14 px), but with ONE
pass of the frozen core per frame (the research code ran it three times) and a backend that can be TensorRT on the Orin.

Backends: "trt" (TensorRT engines built on the device by deploy/build_engines.sh), "onnx" (ONNX Runtime, any machine), "torch" (the PyTorch
checkpoints, reference). Post-processing runs in PyTorch on the same device as the engines (CUDA on the Orin; CPU works).

    from pfm_runtime import GatePerception
    gp = GatePerception("deploy/bundle", backend="trt")
    r = gp(frame_rgb_uint8)            # HxWx3 numpy, or torch (3,H,W)/(1,3,H,W) float 0..1
    r["gates"]  -> (n,9) float: score, x0,y0, x1,y1, x2,y2, x3,y3  (px of the input frame; tl,tr,br,bl for main-head gates)
    r["peaks"]  -> (m,4): type 0..3 (tl,tr,br,bl), score, u, v    (EKF partial-gate update)
    r["seg"]    -> (448,448) uint8 class map (bundle.json seg_classes), r["depth"] -> (448,448) relative inverse depth, r["ms"] stage times
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
try:
    from pfm_refine import refine_gates, sample_patches, to_image, patch_frames, P as P_REF      # vendored copy of src/pfm/refine.py in the device package (deploy/pack.sh): no timm / project import needed
except ImportError:
    sys.path.insert(0, str(ROOT / "src")); from pfm.refine import refine_gates, sample_patches, to_image, patch_frames, P as P_REF      # dev PC: the project module


from gdet_post import gdet_select


# ---------------------------------------------------------------- backends: image (1,3,448,448) 0..1 -> dict of tensors on `dev`
NAMES = ["depth", "seg", "gates", "cpk_heat", "cpk_off", "ng_gates", "pooled"]


class TorchBackend:
    def __init__(self, bundle, dev):
        for k, v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(k, v)
        sys.path.insert(0, str(HERE)); from export_system import load_models, System, RefinerWrap
        core, cpk, ng, ref, segm, segm2, inhead = load_models(); self.sys = System(core, cpk, ng, segm, segm2, inhead).to(dev).eval()
        self.names = NAMES + (["seg_in"] if inhead is not None else []); self.ref = RefinerWrap(ref).to(dev).eval()
    @torch.no_grad()
    def system(self, x): return dict(zip(self.names, self.sys(x)))
    @torch.no_grad()
    def refiner(self, p): return self.ref(p)


class OnnxBackend:
    def __init__(self, bundle, dev):
        import onnxruntime as ort
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.s = ort.InferenceSession(str(bundle / "system_448.onnx"), providers=prov); self.r = ort.InferenceSession(str(bundle / "refiner.onnx"), providers=prov); self.dev = dev
    def system(self, x): return {n.name: torch.from_numpy(o).to(self.dev) for n, o in zip(self.s.get_outputs(), self.s.run(None, {"image": x.float().cpu().numpy()}))}
    def refiner(self, p):
        o = self.r.run(None, {"patches": p.float().cpu().numpy()}); return torch.from_numpy(o[0]).to(self.dev), torch.from_numpy(o[1]).to(self.dev)


class TrtBackend:
    """TensorRT 10 engines; I/O buffers are torch CUDA tensors (no pycuda).
    CUDA graphs (default; PFM_CUDA_GRAPH=0 disables): the static system engine is captured once and replayed per frame; the refiner (dynamic N)
    gets one execution context + graph per batch bucket (patches zero-padded up to the bucket). Measured on the Orin NX (docs/ORIN_RESULTS.md,
    trtexec --useCudaGraph): -12 % (384 px) to -21 % (224 px) per inference, from kernel-launch overhead on the slow A78AE cores."""
    MAXP = 64; BUCKETS = tuple(range(4, 68, 4))      # Update 165: one bucket per gate count (4 patches per gate), no padded patches (refiner engine ~0.24 ms per patch on the Orin)
    def __init__(self, bundle, dev):
        sys.path.append("/usr/lib/python3.12/dist-packages"); import tensorrt as trt
        self.trt = trt; lg = trt.Logger(trt.Logger.WARNING); trt.init_libnvinfer_plugins(lg, ""); rt = trt.Runtime(lg); self.dev = dev   # plugins: m2f head (deformable attention)
        lo, hi = torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, "priority_range") else (0, -1)
        self.stream = torch.cuda.Stream(priority=hi)      # Update 166: system engine (gates / depth) on the HIGH-priority stream
        self.hstream = torch.cuda.Stream(priority=lo)     # heavy async heads (m2f, m2f_out: 47 ms) in the background on a LOW-priority stream
        self.use_graph = os.environ.get("PFM_CUDA_GRAPH", "1") != "0"
        def load(name):
            e = rt.deserialize_cuda_engine((bundle / name).read_bytes()); assert e is not None, f"cannot load {name} (build it with deploy/build_engines.py)"
            return e
        self.se = load(json.loads((bundle / "bundle.json").read_text())["graphs"]["system"].replace(".onnx", ".plan")); self.sc = self.se.create_execution_context(); self.re = load("refiner.plan")
        # two buffer sets (slots) so frame n+1 can run on the GPU while the host post-processes frame n (GatePerception.step); one graph per slot
        self.sbs, self.sgraphs, self.ev = [], [], []
        for slot in range(2):
            sb = {}
            for i in range(self.se.num_io_tensors):
                n = self.se.get_tensor_name(i); dt = {trt.DataType.HALF: torch.float16, trt.DataType.INT32: torch.int32}.get(self.se.get_tensor_dtype(n), torch.float32)
                sb[n] = torch.zeros(tuple(self.se.get_tensor_shape(n)), device=dev, dtype=dt)
            self.sbs.append(sb); self._bind(slot)
            self.sgraphs.append(self._capture(lambda: self.sc.execute_async_v3(self.stream.cuda_stream)) if self.use_graph else None)
            self.ev.append(torch.cuda.Event())
        self.sb = self.sbs[0]; self.sgraph = self.sgraphs[0]
        # split bundles (bundle.json "heads"): head engines read the system engine's tap outputs of the same slot directly (no copy)
        self.heads = {}; cfg = json.loads((bundle / "bundle.json").read_text()); self.last_head = [None, None]
        self.async_names = set(x for x in os.environ.get("PFM_ASYNC_HEADS", "m2f,m2f_out").split(",") if x); self.ahead = {}; self.abusy = None
        self.aev = torch.cuda.Event(); self.acp = [torch.cuda.Event(), torch.cuda.Event()]; self.acp_used = [False, False]
        self.every = dict(cfg.get("head_every") or {})
        for kv in filter(None, os.environ.get("PFM_HEAD_EVERY", "").split(",")): k_, v_ = kv.split("="); self.every[k_] = int(v_)     # e.g. m2f=20,m2f_out=20
        self.every = self.every; self.nlaunch = 0; self.ran = [False, False]; self.seg_in_cache = None; self.seg_fine_cache = None     # heavy heads run every N-th frame (m2f)
        fixed = os.environ.get("PFM_SCENE", "auto")          # scene chosen by the operator: do not even load the other scene's module engine
        skip = {"indoor": {"m2f_out"}, "outdoor": {"m2f"}}.get(fixed, set()) | set(filter(None, os.environ.get("PFM_SKIP_HEADS", "").split(",")))
        for hname, fname in (cfg.get("heads") or {}).items():
            if hname in skip: continue
            fname = os.environ.get(f"PFM_{hname.upper()}_PLAN", fname)          # e.g. PFM_M2F_PLAN=m2f_indoor_288x768_ft_plugin.plan
            he = load(fname.replace(".onnx", ".plan")); per_slot = []
            if hname in self.async_names:          # one context with PRIVATE input buffers (the slot buffers are overwritten while it still runs)
                hc = he.create_execution_context(); ins, outs = {}, {}
                for i in range(he.num_io_tensors):
                    n = he.get_tensor_name(i); dt = {trt.DataType.HALF: torch.float16, trt.DataType.INT32: torch.int32}.get(he.get_tensor_dtype(n), torch.float32)
                    t = torch.zeros(tuple(he.get_tensor_shape(n)), device=dev, dtype=dt); hc.set_tensor_address(n, t.data_ptr())
                    (ins if he.get_tensor_mode(n) == trt.TensorIOMode.INPUT else outs)[n] = t
                g = self._capture(lambda hc=hc: hc.execute_async_v3(self.hstream.cuda_stream), self.hstream) if self.use_graph else None
                self.ahead[hname] = (hc, ins, outs, g); self.heads[hname] = (he, None); continue
            for slot in range(2):
                hc = he.create_execution_context(); outs = {}
                for i in range(he.num_io_tensors):
                    n = he.get_tensor_name(i)
                    if he.get_tensor_mode(n) == trt.TensorIOMode.INPUT: hc.set_tensor_address(n, self.sbs[slot][n].data_ptr())
                    else:
                        dt = {trt.DataType.HALF: torch.float16, trt.DataType.INT32: torch.int32}.get(he.get_tensor_dtype(n), torch.float32)
                        outs[n] = torch.zeros(tuple(he.get_tensor_shape(n)), device=dev, dtype=dt); hc.set_tensor_address(n, outs[n].data_ptr())
                g = self._capture(lambda hc=hc: hc.execute_async_v3(self.stream.cuda_stream)) if self.use_graph else None
                per_slot.append((hc, outs, g))
            self.heads[hname] = (he, per_slot)
        self.taps = {"s4", "s8", "feat", "seg_core", "s16"}
        self.rb = {}                                  # bucket -> (context, in, off, vis, graph)
        for nb in (self.BUCKETS if self.use_graph else (self.MAXP,)):
            ctx = self.re.create_execution_context(); ctx.set_input_shape("patches", (nb, 3, 64, 64))
            bi, bo, bv = torch.zeros(nb, 3, 64, 64, device=dev), torch.zeros(nb, 2, device=dev), torch.zeros(nb, device=dev)
            for name, t in (("patches", bi), ("off", bo), ("vis", bv)): ctx.set_tensor_address(name, t.data_ptr())
            g = self._capture(lambda ctx=ctx: ctx.execute_async_v3(self.stream.cuda_stream)) if self.use_graph else None
            self.rb[nb] = (ctx, bi, bo, bv, g)

    def _capture(self, run, stream=None):
        """Warm up (TensorRT allocates / tunes lazily on the first launches), then capture one launch on `stream` (default self.stream). None if capture fails."""
        st = stream or self.stream
        try:
            with torch.cuda.stream(st):
                for _ in range(3): run()
            st.synchronize(); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st): run()
            st.synchronize(); return g
        except Exception as e:                                                   # noqa: BLE001
            print(f"[pfm_runtime] CUDA graph capture failed ({e}); running without graph", flush=True); torch.cuda.synchronize(); return None

    def _bind(self, slot):
        for n, b in self.sbs[slot].items(): self.sc.set_tensor_address(n, b.data_ptr())

    def launch(self, x, slot=0, head=None, full=None):
        """Enqueue the system engine for x into buffer slot `slot` on self.stream (no host wait). The engine stream first waits for the default stream:
        x (resize) is produced there, and earlier reads of this slot's outputs (post-processing of an older frame) must be finished before it is overwritten."""
        self.stream.wait_stream(torch.cuda.current_stream())
        if self.acp_used[slot]: self.stream.wait_event(self.acp[slot]); self.acp_used[slot] = False      # the async head copied this slot's inputs
        with torch.cuda.stream(self.stream):
            self.sbs[slot]["image"].copy_(x)
            if self.sgraphs[slot] is not None: self.sgraphs[slot].replay()
            else: self._bind(slot); self.sc.execute_async_v3(self.stream.cuda_stream)
            run_head = head is not None and head in self.heads and self.nlaunch % int(self.every.get(head, 1)) == 0
            if run_head and head in self.ahead:                  # background head: start only when the previous run finished; never blocks this stream
                run_head = False
                if self.abusy is None:
                    hc, ins, outs, g = self.ahead[head]; sysdone = torch.cuda.Event(); sysdone.record(self.stream)
                    self.hstream.wait_event(sysdone)
                    with torch.cuda.stream(self.hstream):
                        for n, t in ins.items():
                            src = self.sbs[slot][n]
                            if n == "image" and full is not None: src = full       # Update 167: the module sees the FULL frame at its own aspect (e.g. 384x1024), not the squashed system input
                            t.copy_(src if src.shape == t.shape else F.interpolate(src.float(), size=t.shape[-2:], mode="bilinear", align_corners=False))
                        self.acp[slot].record(self.hstream); self.acp_used[slot] = True
                        if g is not None: g.replay()
                        else: hc.execute_async_v3(self.hstream.cuda_stream)
                        self.aev.record(self.hstream)
                    self.abusy = head
            if run_head:                                          # the chosen plug-in head, right after the system engine on the same stream
                hc, _, g = self.heads[head][1][slot]
                if g is not None: g.replay()
                else: hc.execute_async_v3(self.stream.cuda_stream)
            self.ran[slot] = run_head; self.nlaunch += 1
            self.last_head[slot] = head if head in self.heads else None
            self.ev[slot].record(self.stream)

    def outputs(self, slot=0, host_wait=False):
        """Outputs of the last launch into `slot`, copied on the default stream after the engine finished (own tensors: the slot is reused later)."""
        if host_wait: self.ev[slot].synchronize()
        torch.cuda.current_stream().wait_event(self.ev[slot])
        o = {n: (b if n == "seg" else b.float() if b.dtype == torch.float16 else b.clone()) for n, b in self.sbs[slot].items() if n != "image" and n not in getattr(self, "taps", ())}
        # "seg" (11 x S x S) is not copied (Update 166): it is only read (argmax) in this frame's post-processing on the current stream, which the next
        # launch into this slot waits for (launch: stream.wait_stream(current_stream))
        h = self.last_head[slot] if hasattr(self, "last_head") else None
        if self.abusy is not None and self.aev.query():         # background head finished: refresh its cached map
            torch.cuda.current_stream().wait_event(self.aev); outs = self.ahead[self.abusy][2]
            if "seg_in" in outs: self.seg_in_cache = outs["seg_in"].clone()
            if "seg_fine" in outs: self.seg_fine_cache = outs["seg_fine"].clone()
            self.abusy = None
        if h is not None and h in self.ahead:
            if "seg_in" in self.ahead[h][2] and self.seg_in_cache is not None: o["seg_in"] = self.seg_in_cache
            if "seg_fine" in self.ahead[h][2] and self.seg_fine_cache is not None: o["seg_fine"] = self.seg_fine_cache
        elif h is not None:
            outs = self.heads[h][1][slot][1]
            if "seg_out" in outs: o["seg"] = outs["seg_out"].clone()          # outdoor: v1+v2 seg replaces the v1 seg
            if "seg_in" in outs:                                               # indoor: 22-class map (heavy heads: last computed map in between)
                if self.ran[slot]: self.seg_in_cache = outs["seg_in"].clone()
                if self.seg_in_cache is not None: o["seg_in"] = self.seg_in_cache
            if "seg_fine" in outs:                                             # outdoor: 19-class Cityscapes-id map (m2f_out head)
                if self.ran[slot]: self.seg_fine_cache = outs["seg_fine"].clone()
                if self.seg_fine_cache is not None: o["seg_fine"] = self.seg_fine_cache
        o["head"] = h
        return o

    def system(self, x, head=None, full=None):
        self.launch(x, 0, head, full); return self.outputs(0, host_wait=True)

    def refiner(self, p):
        outs_off, outs_vis = [], []
        for i in range(0, len(p), self.MAXP):
            q = p[i:i + self.MAXP]; n = len(q); nb = next(b for b in self.rb if b >= n); ctx, bi, bo, bv, g = self.rb[nb]
            self.stream.wait_stream(torch.cuda.current_stream())    # patches come from the default stream (same race as system())
            with torch.cuda.stream(self.stream):
                bi[:n].copy_(q)
                if nb > n: bi[n:].zero_()
                if g is not None: g.replay()
                else: ctx.set_input_shape("patches", (n, 3, 64, 64)); ctx.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize(); outs_off.append(bo[:n].clone()); outs_vis.append(bv[:n].clone())
        return torch.cat(outs_off), torch.cat(outs_vis)


BACKENDS = dict(torch=TorchBackend, onnx=OnnxBackend, trt=TrtBackend)


# ---------------------------------------------------------------- post-processing (identical rules to ours_detector.py, min14 config)
def order_tl_tr_br_bl(p):
    c = p.mean(0); ang = torch.atan2(p[:, 1] - c[1], p[:, 0] - c[0]); p = p[ang.argsort()]; s = (p[:, 0] + p[:, 1]).argmin()
    return torch.roll(p, -int(s), 0)


def decode_peaks(heat, off, out_hw, thr, k=3, max_per_type=6):
    hs = heat.float().sigmoid()[0]; H, W = hs.shape[-2:]; pk = (F.max_pool2d(hs[None], k, 1, k // 2)[0] == hs) & (hs > thr); sy, sx = out_hw[0] / H, out_hw[1] / W; rows = []
    for t in range(4):
        yx = pk[t].nonzero()
        if len(yx) == 0: continue
        sc = hs[t, yx[:, 0], yx[:, 1]]; o = sc.argsort(descending=True)[:max_per_type]; yx, sc = yx[o], sc[o]
        u = (yx[:, 1].float() + 0.5 + off[0, 0, yx[:, 0], yx[:, 1]]) * sx; v = (yx[:, 0].float() + 0.5 + off[0, 1, yx[:, 0], yx[:, 1]]) * sy
        rows.append(torch.stack((torch.full_like(sc, t), sc, u, v), -1))
    return torch.cat(rows) if rows else torch.zeros(0, 4, device=heat.device)


def peaks_from(d):
    if len(d) == 0: return torch.zeros(0, 4, device=d.device)
    c = d[:, 1:].view(-1, 4, 2); sc = d[:, 0:1].expand(-1, 4); t = torch.arange(4, device=d.device, dtype=torch.float32)[None].expand(len(d), -1)
    return torch.stack((t, sc, c[..., 0], c[..., 1]), -1).reshape(-1, 4)


def dedup(d, iou_thr, inside_rule):
    if len(d) < 2: return d
    c = d[:, 1:].view(-1, 4, 2); lo, hi = c.amin(1), c.amax(1); ctr = c.mean(1); area = (hi - lo).clamp(min=1e-3).prod(-1); keep = []
    for i in d[:, 0].argsort(descending=True).tolist():
        ok = True
        for j in keep:
            inter = (torch.minimum(hi[i], hi[j]) - torch.maximum(lo[i], lo[j])).clamp(min=0).prod(); iou = inter / (area[i] + area[j] - inter)
            inside = bool(((ctr[i] >= lo[j]) & (ctr[i] <= hi[j])).all() or ((ctr[j] >= lo[i]) & (ctr[j] <= hi[i])).all())
            if iou > iou_thr or (inside_rule and inside and iou > 0.05): ok = False; break
        if ok: keep.append(i)
    return d[sorted(keep)]


def sample_patches_1img(img, corner, centroid, size):
    """= pfm_refine.sample_patches for one image, without img[bidx] (that copied the whole camera frame once per corner: ~12 ms on the Orin
    for 16 corners at 1640x1232). All patches come from ONE grid_sample call on the (1,3,H,W) frame."""
    _, _, H, W = img.shape; n = len(corner)
    s, R = patch_frames(corner, centroid, size)
    t = torch.arange(P_REF, device=img.device, dtype=torch.float32) - (P_REF - 1) / 2
    qy, qx = torch.meshgrid(t, t, indexing="ij"); q = torch.stack([qx, qy], -1).view(1, P_REF * P_REF, 2)
    pts = corner[:, None] + s[:, None, None] * torch.einsum("nij,nkj->nki", R, q.expand(n, -1, -1))
    grid = torch.stack([(pts[..., 0] + 0.5) / W * 2 - 1, (pts[..., 1] + 0.5) / H * 2 - 1], -1).view(1, n * P_REF, P_REF, 2)
    out = F.grid_sample(img[:1].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)          # (1,3,n*P,P)
    return out.view(3, n, P_REF, P_REF).permute(1, 0, 2, 3).contiguous(), s, R


def refine_vec(refiner_fn, img, g, margin, vis_thr):
    """Vectorised refine_gates for one image: g (k,8) corner px on img's device -> refined (k,8). Same rules: corners beyond `margin` px
    outside the image and corners with visibility logit < vis_thr keep the coarse value."""
    H, W = img.shape[-2:]; c = g.view(-1, 4, 2).float(); ctr = c.mean(1, keepdim=True).expand(-1, 4, -1); sz = (c.amax(1) - c.amin(1)).amax(-1)[:, None].expand(-1, 4)
    m = (c[..., 0] >= -margin) & (c[..., 0] < W + margin) & (c[..., 1] >= -margin) & (c[..., 1] < H + margin)
    if not bool(m.any()): return g
    cor, cen, size = c[m], ctr[m], sz[m]
    patches, s_, R = sample_patches_1img(img, cor, cen, size)
    off, vis = refiner_fn(patches); new = to_image(off.float(), cor, s_, R)
    out = c.clone(); upd = out[m]; ok = vis >= vis_thr; upd[ok] = new[ok]; out[m] = upd
    return out.reshape(-1, 8)


# ---------------------------------------------------------------- fast path (default): same rules, numpy on the CPU, ONE device->host copy
# The torch path above pays ~5-10 us framework overhead per tiny op and several GPU syncs; on the Orin's A78AE cores that was ~10 ms/frame.
def gpu_peaks(heat, off, img, out_hw, thr, k=3, max_per_type=6):
    """decode_peaks + the derotation-border filter, fully on the device: -> (4*max_per_type, 5) type, score, u, v, keep (score <= thr or
    border peaks have keep 0). top-k per type == argsort(desc)[:max_per_type] of decode_peaks."""
    hs = heat.float().sigmoid()[0]; T, H, W = hs.shape; pk = (F.max_pool2d(hs[None], k, 1, k // 2)[0] == hs) & (hs > thr)
    sc, idx = torch.where(pk, hs, torch.full_like(hs, -1.0)).view(T, -1).topk(min(max_per_type, H * W), dim=1)
    yy, xx = idx // W, idx % W; o = off[0].float().view(2, -1)
    u = (xx.float() + 0.5 + o[0][idx]) * (out_hw[1] / W); v = (yy.float() + 0.5 + o[1][idx]) * (out_hw[0] / H)
    h, w = out_hw                                                                        # zero-filled borders after derotation
    r = torch.arange(-8, 9, device=img.device)
    uu = u.long().clamp(0, w - 1).view(-1); vv = v.long().clamp(0, h - 1).view(-1)
    win = img[0][:, (vv[:, None, None] + r[None, :, None]).clamp(0, h - 1), (uu[:, None, None] + r[None, None, :]).clamp(0, w - 1)].amax(0) > 1e-3   # (P,17,17), only the windows
    keep = win.flatten(1).all(1).view(T, -1) & (sc > thr)
    t = torch.arange(T, device=img.device, dtype=torch.float32)[:, None].expand_as(sc)
    return torch.stack((t, sc, u, v, keep.float()), -1).view(-1, 5)


def order_np(p):
    c = p.mean(0); p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]), kind="stable")]; return np.roll(p, -int((p[:, 0] + p[:, 1]).argmin()), 0)


def dedup_np(d, iou_thr, inside_rule):
    if len(d) < 2: return d
    c = d[:, 1:].reshape(-1, 4, 2); lo, hi = c.min(1), c.max(1); ctr = c.mean(1); area = np.clip(hi - lo, 1e-3, None).prod(-1); keep = []
    for i in np.argsort(-d[:, 0], kind="stable"):
        ok = True
        for j in keep:
            inter = np.clip(np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j]), 0, None).prod(); iou = inter / (area[i] + area[j] - inter)
            inside = bool(((ctr[i] >= lo[j]) & (ctr[i] <= hi[j])).all() or ((ctr[j] >= lo[i]) & (ctr[j] <= hi[i])).all())
            if iou > iou_thr or (inside_rule and inside and iou > 0.05): ok = False; break
        if ok: keep.append(i)
    return d[sorted(keep)]


def peaks_from_np(d):
    if len(d) == 0: return np.zeros((0, 4), np.float32)
    c = d[:, 1:].reshape(-1, 4, 2); n = len(d)
    return np.stack((np.tile(np.arange(4, dtype=np.float32), (n, 1)), np.repeat(d[:, :1], 4, 1), c[..., 0], c[..., 1]), -1).reshape(-1, 4)


def _sig(x): return 1.0 / (1.0 + np.exp(-x))


class _RefinerShim:
    """refine_gates() calls model(patches) -> (off, vis, heat); the backend gives (off, vis)."""
    def __init__(self, fn): self.fn = fn
    def __call__(self, p): off, vis = self.fn(p); return off, vis, None


class GatePerception:
    def __init__(self, bundle="deploy/bundle", backend="trt", device=None, router=None, modules=None):
        self.bundle = Path(bundle); self.cfg = json.loads((self.bundle / "bundle.json").read_text()); self.P = self.cfg["post"]; self.S = self.cfg["input_size"]
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.be = BACKENDS[backend](self.bundle, self.dev); self.backend = backend
        self.ref = _RefinerShim(self.be.refiner)
        self.router = None      # optional condition router (pooled feature LDA); modules are input-side adapters -> second system pass (off by default: none accepted on real data, Update 149)
        self.fast = os.environ.get("PFM_POST", "fast") != "torch"      # PFM_POST=torch: the reference torch post-processing
        self.bgr = False; self.pre = None; self.last_img = None
        # CUDA graphs for the small post-processing ops (TensorRT backend): on the Orin each PyTorch launch costs ~0.1-0.2 ms of CPU, gpu_peaks and the
        # refiner chain (patch sampling -> engine -> read-out) are ~40 launches; replayed as graphs they are one launch each. PFM_POST_GRAPH=0 disables.
        self.graphs = backend == "trt" and os.environ.get("PFM_POST_GRAPH", "1") != "0" and self.dev != "cpu"; self.img_bufs = [None, None]; self._g = {}
        self._slot = 0; self._pending = None; self._cur_slot = 0
        # scene router (split bundles): logistic regression on the pooled feature, switch after `router_hold` consecutive frames; PFM_SCENE=indoor|outdoor forces
        rc = self.cfg.get("router"); self.router_w = None; self.scene = os.environ.get("PFM_SCENE", "outdoor" if rc else None); self._votes = 0
        self.scene_auto = rc is not None and os.environ.get("PFM_SCENE", "auto") == "auto"; self.p_indoor = None
        if rc: self.router_w = (np.array(rc["mean"], np.float32), np.array(rc["std"], np.float32), np.array(rc["w"], np.float32), float(rc["b"])); self.hold = int(self.cfg.get("router_hold", 3))

    def _head(self):
        if self.scene is None or not getattr(self.be, "heads", None): return None
        if self.scene == "indoor": return "m2f" if "m2f" in self.be.heads else "indoor"
        return "m2f_out" if "m2f_out" in self.be.heads else "seg2"

    # Update 167: the 11-class map from the indoor / outdoor module (much better labels than the frozen core's seg head), core kept for "gate"
    # (and lane marking) pixels. PFM_SEG_FROM_MODULE=0 = core seg only. Cityscapes 19 -> 11 and indoor 22 -> 11 class lookup tables:
    LUT_OUT = [4, 4, 5, 5, 10, 2, 2, 2, 2, 4, 0, 3, 3, 7, 7, 7, 7, 7, 10]
    LUT_IN = [10, 5, 4, 5, 5, 5, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2, 10, 10]

    def _seg11(self, o):
        core = o["seg"][0].argmax(0).to(torch.uint8)
        if os.environ.get("PFM_SEG_FROM_MODULE", "1") != "1": return core
        for key, lut in (("seg_fine", self.LUT_OUT), ("seg_in", self.LUT_IN)):
            if key in o and o[key] is not None:
                if getattr(self, "_lut_t", None) is None: self._lut_t = {}
                t = self._lut_t.get(key)
                if t is None: t = self._lut_t[key] = torch.tensor(lut, dtype=torch.uint8, device=core.device)
                mm = o[key][0].long().clamp_(0, len(lut) - 1)
                mm = F.interpolate(t[mm][None, None].float(), size=core.shape, mode="nearest")[0, 0].to(torch.uint8)
                keep = (core == 1) | (core == 9)                      # gate / lane marking stay from the core
                return torch.where(keep, core, mm)
        return core

    def _route(self, o):
        """Update the scene from this frame's pooled feature (used for the NEXT frame's head: no wait on the current GPU work)."""
        if self.router_w is None or not self.scene_auto: return          # fixed scene: no per-frame GPU->CPU copy of the pooled feature
        mu, sd, w, b = self.router_w; z = (o["pooled"].reshape(-1).float().cpu().numpy() - mu) / sd; self.p_indoor = float(1 / (1 + np.exp(-(z @ w + b))))
        if not self.scene_auto: return
        want = "indoor" if self.p_indoor > 0.5 else "outdoor"
        self._votes = self._votes + 1 if want != self.scene else 0
        if self._votes >= self.hold: self.scene, self._votes = want, 0

    def _select(self, o):
        """Bundles with candidate outputs (gates_format in bundle.json): gate-head selection on the host (deploy/gdet_post.py), in place -> (1,K,9)."""
        if o["gates"].shape[-1] == 9: return
        g = self.cfg["gdet"]; c = torch.cat((o["gates"].reshape(-1), o["ng_gates"].reshape(-1))).cpu().numpy().reshape(2, o["gates"].shape[1], -1)
        o["gates"] = torch.from_numpy(gdet_select(c[0], g["k"], g["main_levels"], g["dedup"], g["merge"]))[None].to(self.dev)
        o["ng_gates"] = torch.from_numpy(gdet_select(c[1], g["k"], g["ng_levels"], g["dedup"], g["merge"]))[None].to(self.dev)

    def _prep(self, frame, slot=0):
        """Upload + optional GPU pre-processing: self.bgr (camera frame in BGR order), self.pre (callable on (1,3,H,W) 0..1, e.g. camera.Undistort.gpu,
        square crop). The detections are in px of the image after self.pre; self.last_img keeps that image (recording)."""
        # Update 165 (nsys on the Orin): the frame stays in the camera's HWC layout (an NCHW view with channels-last strides) all the way; the old
        # path made the persistent buffer NCHW-contiguous, i.e. a full-frame transpose copy (2.9 ms/frame at 1640x1232) plus separate float and /255
        # passes (1.4 ms). Same values, the graphs / grid_sample / interpolate read the strided view.
        if isinstance(frame, np.ndarray) and frame.dtype == np.uint16:      # raw Bayer (v4l2raw.RawCamera)
            return self._prep_raw(frame, slot)
        if isinstance(frame, np.ndarray) and frame.ndim == 2 and getattr(self, "nv12_grid", None) is not None:
            return self._prep_nv12_remap(frame, slot)
        if isinstance(frame, np.ndarray) and frame.ndim == 2:      # NV12 camera buffer (camera.GstPipe nv12): (H*3/2, W) -> RGB on the GPU, BT.601 video range
            u = torch.from_numpy(frame).to(self.dev); H = u.shape[0] * 2 // 3; W = u.shape[1]
            y = u[:H].float() - 16; uv = u[H:].view(H // 2, W // 2, 2).float() - 128
            uv = uv.repeat_interleave(2, 0).repeat_interleave(2, 1); cb, cr = uv[..., 0], uv[..., 1]
            x = torch.stack((1.164 * y + 1.596 * cr, 1.164 * y - 0.392 * cb - 0.813 * cr, 1.164 * y + 2.017 * cb))[None].mul_(1.0 / 255).clamp_(0, 1)
            if self.pre is not None: x = self.pre(x)
            if self.graphs:
                b = self.img_bufs[slot]
                if b is None or b.shape != x.shape or b.stride() != x.stride(): b = self.img_bufs[slot] = torch.empty_like(x); self._g = {k: v for k, v in self._g.items() if k[-1] != slot}
                b.copy_(x); x = b
            self.last_img = x
            return x
        if isinstance(frame, np.ndarray):
            u = torch.from_numpy(frame).to(self.dev)
            if u.shape[-1] == 4: u = u[..., :3]          # BGRx / RGBA camera buffer
            if self.bgr: u = u.flip(-1)
            u = u.permute(2, 0, 1)[None]                 # uint8 NCHW view, channels-last strides
            if self.pre is None and self.graphs:          # straight into the persistent float buffer: one conversion pass + one in-place scale
                b = self.img_bufs[slot]
                if b is None or b.shape != u.shape: b = self.img_bufs[slot] = torch.empty_like(u, dtype=torch.float32); self._g = {k: v for k, v in self._g.items() if k[-1] != slot}
                b.copy_(u); b.mul_(1.0 / 255); self.last_img = b
                return b
            x = u.float().mul_(1.0 / 255)
        else:
            x = frame.to(self.dev).float(); x = x[None] if x.dim() == 3 else x
            if x.shape[1] == 4: x = x[:, :3]
            if self.bgr: x = x.flip(1)
        if self.pre is not None: x = self.pre(x)
        if self.graphs:              # persistent frame buffers (one per slot): the captured graphs read the frame from a fixed address
            b = self.img_bufs[slot]
            if b is None or b.shape != x.shape or b.stride() != x.stride(): b = self.img_bufs[slot] = torch.empty_like(x); self._g = {k: v for k, v in self._g.items() if k[-1] != slot}
            b.copy_(x); x = b
        self.last_img = x
        return x

    def _prep_raw(self, frame, slot):
        """Raw RGGB Bayer frame from v4l2raw.RawCamera (uint16, 10-bit << 6, black ~4096) -> RGB on the GPU (Update 167): black level,
        Malvar-He-Cutler 5x5 demosaic (bilinear gave zipper / comb artefacts on edges), grey-world white balance (EMA), colour-correction
        matrix (raw sensor colours are washed out; the NVIDIA ISP applies one too), chroma denoise (5x5 box on the colour difference only),
        digital gain to an 18 % grey level, sRGB-like gamma, then the optional undistort grid (self.raw_grid) and rotation.
        self.raw_stats keeps the linear grey level for the sensor auto exposure (run.py)."""
        H, W = frame.shape
        if getattr(self, "_cuda_isp", None) is None and os.environ.get("PFM_ISP", "cuda") == "cuda":      # 3.4 ms on the Orin vs 25 ms (deploy/isp_cuda.py)
            try:
                os.environ.setdefault("CUDA_HOME", "/usr/local/cuda"); os.environ["PATH"] = f"{Path(sys.executable).parent}:/usr/local/cuda/bin:" + os.environ.get("PATH", "")
                import isp_cuda; self._cuda_isp = isp_cuda.CudaISP(self.dev)
            except Exception as e:                                                       # noqa: BLE001
                print(f"[pfm_runtime] CUDA ISP unavailable ({e}); using the PyTorch ISP", flush=True); self._cuda_isp = False
        if self._cuda_isp if getattr(self, "_cuda_isp", None) is not None else False:
            isp = self._cuda_isp; isp.row_shift = getattr(self, "raw_row_shift", None); rgb = isp(frame); self.raw_stats = isp.lin
            gr = getattr(self, "raw_grid", None)
            if gr is not None: rgb = F.grid_sample(rgb, gr, mode="bilinear", padding_mode="zeros", align_corners=False)
            rot = getattr(self, "raw_rotate", 0)
            if rot: rgb = torch.rot90(rgb, k=-rot // 90, dims=(2, 3))
            oh, ow = rgb.shape[-2:]; self.full_hw = (oh, ow)
            bb = self.img_bufs[slot]
            if bb is None or tuple(bb.shape[-2:]) != (oh, ow): bb = self.img_bufs[slot] = torch.empty(1, 3, oh, ow, device=self.dev); self._g = {k_: v for k_, v in self._g.items() if k_[-1] != slot}
            bb.copy_(rgb); self.last_img = bb
            return bb
        u = torch.from_numpy(frame.view(np.int16)).to(self.dev, non_blocking=True).to(torch.int32).bitwise_and_(0xFFFF)
        if getattr(self, "_rawc", None) is None or self._rawc[0] != (H, W):
            d = self.dev; m = torch.zeros(4, H, W, device=d)
            m[0, 0::2, 0::2] = 1; m[1, 0::2, 1::2] = 1; m[2, 1::2, 0::2] = 1; m[3, 1::2, 1::2] = 1          # R, Gr (R row), Gb (B row), B
            kG = torch.tensor([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype=torch.float32) / 8
            kRow = torch.tensor([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]], dtype=torch.float32) / 8
            kCol = kRow.t().contiguous()
            kX = torch.tensor([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], dtype=torch.float32) / 8
            k = torch.stack((kG, kRow, kCol, kX))[:, None].to(d)                                          # (4,1,5,5) Malvar-He-Cutler
            ccm = torch.tensor([[1.70, -0.50, -0.20], [-0.35, 1.60, -0.25], [-0.05, -0.55, 1.60]], device=d)    # rows sum to 1 (greys stay grey)
            self._rawc = ((H, W), m, k, ccm); self.wb = torch.ones(3, device=d); self.dgain = torch.ones((), device=d)
        _, m, k, ccm = self._rawc
        x = (u.float() - 4096.0).clamp_(min=0).mul_(1.0 / (65535.0 - 4096.0))                           # (H,W) linear mosaic
        sh = getattr(self, "raw_row_shift", 8 if W == 1640 else 0)
        if sh:   # IMX219 x2-binned full-width readout (x start 0): odd (G/B) rows come out 8 px LEFT of the even (R/G) rows -> double edges and
                 # colour fringes (measured on the factory 1640x1232 mode too: plane shift +3.79 -> -0.21 after this, = the aligned 1280x720 mode)
            if sh:   # row y comes out sh*(y%4) px LEFT (x2-binned full-width readout, factory mode too): shift classes 1,2,3 by sh, 2sh, 3sh
                x = x.clone()
                for c_ in (1, 2, 3): d_ = sh * c_; x[c_::4, d_:] = x[c_::4, :-d_].clone(); x[c_::4, :d_] = x[c_::4, d_:d_ + 1]
        c = F.conv2d(F.pad(x[None, None], (2, 2, 2, 2), mode="reflect"), k)[0]                          # (4,H,W): G@RB, row-neighbour, col-neighbour, cross
        R_, Gr, Gb, B_ = m[0], m[1], m[2], m[3]
        r = x * R_ + c[1] * Gr + c[2] * Gb + c[3] * B_
        g = x * (Gr + Gb) + c[0] * (R_ + B_)
        b = x * B_ + c[1] * Gb + c[2] * Gr + c[3] * R_
        rgb = torch.stack((r, g, b))[None].clamp_(min=0)                                                # (1,3,H,W) linear
        mean = rgb[0, :, ::8, ::8].mean((1, 2))
        self.wb.mul_(0.9).add_(0.1 * (mean[1] / mean.clamp(min=1e-4)).clamp(0.4, 3.0))                  # grey world, green = 1
        lin = (mean * self.wb).mean(); self.raw_stats = lin
        self.dgain.mul_(0.9).add_(0.1 * (0.18 / lin.clamp(min=1e-4)).clamp(0.5, 16.0))          # up to 16x (dim scenes at a 2 ms exposure)
        rgb = rgb * (self.wb * self.dgain).view(1, 3, 1, 1)
        rgb = torch.einsum("ij,bjhw->bihw", ccm, rgb)                                                  # colour correction
        yy = rgb.mean(1, keepdim=True); rgb = yy + F.avg_pool2d(rgb - yy, 5, 1, 2, count_include_pad=False)   # chroma denoise, luma untouched
        rgb = rgb.clamp_(0, 1).pow_(1 / 2.2)
        # comb / row-alternation of the x2-binned readout: vertical [1,2,1]/4 (measured: comb 38 -> 15, noise 47 -> 26, sharpness 32 -> 20),
        # then a light edge-preserving luma denoise (guided filter on luma, r 1, eps 5e-4)
        rgb = torch.cat((rgb[:, :, :1], 0.25 * rgb[:, :, :-2] + 0.5 * rgb[:, :, 1:-1] + 0.25 * rgb[:, :, 2:], rgb[:, :, -1:]), 2)
        Y = rgb.mean(1, keepdim=True); mY = F.avg_pool2d(Y, 3, 1, 1, count_include_pad=False); vY = F.avg_pool2d(Y * Y, 3, 1, 1, count_include_pad=False) - mY * mY
        A = vY / (vY + 5e-4); Bc = mY - A * mY; Yn = F.avg_pool2d(A, 3, 1, 1, count_include_pad=False) * Y + F.avg_pool2d(Bc, 3, 1, 1, count_include_pad=False)
        rgb = (rgb + (Yn - Y)).clamp_(0, 1)
        gr = getattr(self, "raw_grid", None)
        if gr is not None: rgb = F.grid_sample(rgb, gr, mode="bilinear", padding_mode="zeros", align_corners=False)
        rot = getattr(self, "raw_rotate", 0)
        if rot: rgb = torch.rot90(rgb, k=-rot // 90, dims=(2, 3))        # camera mounted rotated: clockwise degrees (run.py --rotate)
        oh, ow = rgb.shape[-2:]; self.full_hw = (oh, ow)
        bb = self.img_bufs[slot]
        if bb is None or tuple(bb.shape[-2:]) != (oh, ow): bb = self.img_bufs[slot] = torch.empty(1, 3, oh, ow, device=self.dev); self._g = {k_: v for k_, v in self._g.items() if k_[-1] != slot}
        bb.copy_(rgb); self.last_img = bb
        return bb

    def _prep_nv12_remap(self, frame, slot):
        """NV12 frame + undistort grid in one pass (Update 166; Orin, 1280x720: 1.96 ms vs 4.12 ms for conversion-then-remap). Y - 16 and UV - 128
        are resampled with the SAME normalised grid (the UV plane is half resolution; normalised coordinates make that free), so the zero padding
        outside the source frame becomes RGB 0 exactly as with the RGB remap (the peak border filter relies on it); BT.601 video range, written
        straight into the persistent frame buffer channel by channel (a 3x3 matmul / 1x1 conv over the frame was 1.6-1.9 ms on the Orin)."""
        u = torch.from_numpy(frame).to(self.dev, non_blocking=True); H = u.shape[0] * 2 // 3; W = u.shape[1]; g = self.nv12_grid
        yy = u[:H].view(1, 1, H, W).float().sub_(16).mul_(1.164 / 255)
        cc = u[H:].view(1, H // 2, W // 2, 2).permute(0, 3, 1, 2).float().sub_(128).mul_(1.0 / 255)
        Y = F.grid_sample(yy, g, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
        C = F.grid_sample(cc, g, mode="bilinear", padding_mode="zeros", align_corners=False)[0]; Cb, Cr = C[0], C[1]
        oh, ow = Y.shape; self.full_hw = (oh, ow)
        b = self.img_bufs[slot]
        if b is None or tuple(b.shape[-2:]) != (oh, ow): b = self.img_bufs[slot] = torch.empty(1, 3, oh, ow, device=self.dev); self._g = {k_: v for k_, v in self._g.items() if k_[-1] != slot}
        torch.add(Y, Cr, alpha=1.596, out=b[0, 0]); torch.add(Y, Cb, alpha=2.017, out=b[0, 2])
        torch.add(Y, Cb, alpha=-0.392, out=b[0, 1]); b[0, 1].add_(Cr, alpha=-0.813); b.clamp_(0, 1)
        self.last_img = b
        return b

    @torch.no_grad()
    def step(self, frame, meta=None):
        """Pipelined call (TensorRT backend): starts frame n on the GPU, then post-processes frame n-1 on the host while the engine runs.
        Returns (result, meta) of the PREVIOUS frame (None for the first call). Same results as __call__ frame by frame (tested); one frame more
        latency, throughput = max(GPU time, host time) instead of their sum."""
        t0 = time.perf_counter(); slot = self._slot; self._slot ^= 1
        img = self._prep(frame, slot); x = F.interpolate(img, size=(self.S, self.S), mode="bilinear", align_corners=False)
        self.be.launch(x, slot, self._head(), full=img if getattr(self.be, "ahead", None) else None)
        prev, self._pending = self._pending, (slot, img, meta, t0)
        return self._finish(prev) if prev is not None else None

    def _finish(self, prev):
        ps, pimg, pmeta, pt0 = prev; self._cur_slot = ps; t1 = time.perf_counter()
        o = self.be.outputs(ps, host_wait=True); self._route(o); self._select(o); t2 = time.perf_counter(); h, w = pimg.shape[-2:]
        r = self._post_fast(o, pimg, h, w, t1, t1, t2); r["ms"]["frame_age"] = 1e3 * (time.perf_counter() - pt0); self.last_img = pimg
        return r, pmeta

    @torch.no_grad()
    def flush(self):
        """Result of the frame still in flight (end of a run)."""
        prev, self._pending = self._pending, None
        return self._finish(prev) if prev is not None else None

    def _capture(self, fn):
        """Capture fn() (torch ops + TensorRT launches on the current stream) as a CUDA graph; returns (graph, outputs)."""
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2): fn()                                   # warm-up (allocator, TensorRT lazy init)
        torch.cuda.current_stream().wait_stream(st); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = fn()
        return g, out

    def _peaks(self, o, img, h, w):
        k = ("pk", tuple(o["cpk_heat"].shape), tuple(img.shape), self._cur_slot)
        if k not in self._g:
            sh, so = torch.empty_like(o["cpk_heat"], dtype=torch.float32), torch.empty_like(o["cpk_off"], dtype=torch.float32)
            sh.copy_(o["cpk_heat"]); so.copy_(o["cpk_off"])
            g, out = self._capture(lambda: gpu_peaks(sh, so, img, (h, w), self.P["cpk_thr"])); self._g[k] = (g, sh, so, out)
        g, sh, so, out = self._g[k]; sh.copy_(o["cpk_heat"]); so.copy_(o["cpk_off"]); g.replay(); return out

    def _refine(self, img, g8):
        """= refine_vec(self.be.refiner, img, g8, ...) with the whole chain replayed as one CUDA graph per gate-count bucket. g8 numpy (k,8) -> numpy (k,8).
        All 4k corners are refined in the graph (the refiner is per-patch, so extra rows do not change the others); the host keeps the update only where
        refine_vec would (corner within `margin` of the image, visibility logit >= vis_thr)."""
        P = self.P; H, W = img.shape[-2:]; k = len(g8); margin, vt = P["refiner_margin"], P["refiner_vis_thr"]
        nb = k if 1 <= k <= 16 else None                      # exact gate-count bucket (was 1/2/4/8/16: 3 gates ran 16 patches)
        if nb is None: return refine_vec(self.be.refiner, img, torch.from_numpy(np.ascontiguousarray(g8)).to(self.dev), margin, vt).cpu().numpy()
        c = g8.reshape(k, 4, 2).astype(np.float32)
        m = (c[..., 0] >= -margin) & (c[..., 0] < W + margin) & (c[..., 1] >= -margin) & (c[..., 1] < H + margin)
        if not m.any(): return g8
        key = ("ref", nb, tuple(img.shape), self._cur_slot)
        if key not in self._g:
            gin = torch.zeros(nb, 4, 2, device=self.dev); ctx, bi, bo, bv, _ = self.be.rb[4 * nb]
            def fn():
                cc = gin.reshape(-1, 2); ctr = gin.mean(1, keepdim=True).expand(-1, 4, -1).reshape(-1, 2); sz = (gin.amax(1) - gin.amin(1)).amax(-1)[:, None].expand(-1, 4).reshape(-1)
                patches, s_, R = sample_patches_1img(img, cc, ctr, sz); bi.copy_(patches); ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                return to_image(bo.float(), cc, s_, R), bv
            gin.copy_(torch.from_numpy(np.repeat(c[:1], nb, 0)))
            g, out = self._capture(fn); self._g[key] = (g, gin, out)
        g, gin, (new, vis) = self._g[key]
        pad = np.concatenate([c, np.repeat(c[:1], nb - k, 0)]) if nb > k else c
        gin.copy_(torch.from_numpy(pad)); g.replay()
        r = torch.cat((new.reshape(-1), vis.reshape(-1))).cpu().numpy(); new_np = r[:8 * nb].reshape(nb, 4, 2)[:k]; vis_np = r[8 * nb:].reshape(nb, 4)[:k]
        upd = m & (vis_np >= vt); out = c.copy(); out[upd] = new_np[upd]; return out.reshape(k, 8)

    @torch.no_grad()
    def __call__(self, frame):
        P = self.P; t0 = time.perf_counter(); self._cur_slot = 0; img = self._prep(frame, 0); h, w = img.shape[-2:]
        x = F.interpolate(img, size=(self.S, self.S), mode="bilinear", align_corners=False); t1 = time.perf_counter()
        o = self.be.system(x, self._head(), full=img if getattr(self.be, "ahead", None) else None) if self.backend == "trt" else self.be.system(x); self._route(o); self._select(o); t2 = time.perf_counter()
        if self.fast: return self._post_fast(o, img, h, w, t0, t1, t2)
        cpu = torch.device("cpu"); wh = torch.tensor([w, h], dtype=torch.float32)
        G = o["gates"][0].float().cpu(); N2 = o["ng_gates"][0].float().cpu()          # tiny (K,9) tensors: post-processing on the CPU avoids per-element GPU syncs
        pk = decode_peaks(o["cpk_heat"], o["cpk_off"], (h, w), P["cpk_thr"])
        if len(pk):         # drop peaks within 8 px of zero-filled (derotation) borders
            valid = -F.max_pool2d(-(img.amax(1, keepdim=True) > 1e-3).float(), 17, 1, 8)
            u = pk[:, 2].long().clamp(0, w - 1); v = pk[:, 3].long().clamp(0, h - 1); pk = pk[valid[0, 0, v, u] > 0.5]
        pk = pk.cpu()
        # 1. main dense head gates (score > thr), geometric corner order
        sc = G[:, 0].sigmoid(); rows = [torch.cat((sc[q:q + 1], order_tl_tr_br_bl(G[q, 1:].view(4, 2) * wh).reshape(-1))) for q in range(len(G)) if sc[q] > P["thr"]]
        d = torch.stack(rows) if rows else torch.zeros(0, 9)
        # 2. near-gate head: add big confident ones, replace a duplicate main detection when larger (corners in simulator identity order)
        nsc = N2[:, 0].sigmoid(); keep = nsc >= P["ng_thr"]; c = N2[keep, 1:].view(-1, 4, 2) * wh; nsc = nsc[keep]
        if len(c):
            size = (c.amax(1) - c.amin(1)).amax(-1); big = size >= P["ng_min"] * max(h, w); c, nsc, size = c[big], nsc[big], size[big]
        for k in range(len(c)):
            row = torch.cat((nsc[k:k + 1], c[k].reshape(-1)))
            if len(d):
                dc = d[:, 1:].view(-1, 4, 2)
                dist = torch.stack([(dc - torch.roll(c[k], r, 0)[None]).norm(dim=-1).mean(-1) for r in range(4)] + [(dc - torch.roll(c[k].flip(0), r, 0)[None]).norm(dim=-1).mean(-1) for r in range(4)]).amin(0)
                j = int(dist.argmin())
                if dist[j] < 0.25 * size[k]:
                    dsz = (dc[j].amax(0) - dc[j].amin(0)).max()
                    if size[k] >= dsz and size[k] >= P["ng_min"] * max(h, w) and nsc[k] >= P["ng_thr"]: d = torch.cat((d[:j], torch.cat((torch.maximum(d[j, :1], nsc[k:k + 1]), c[k].reshape(-1)))[None], d[j + 1:]))
                    continue
            d = torch.cat((d, row[None])) if len(d) else row[None]
        t3 = time.perf_counter()
        # 3. corner refiner (local patches around every corner, sampled on the GPU)
        if len(d): d = torch.cat((d[:, :1], refine_vec(self.be.refiner, img, d[:, 1:].to(self.dev), P["refiner_margin"], P["refiner_vis_thr"]).cpu()), 1)
        t4 = time.perf_counter()
        # 4. dedup + min size
        if P["dedup_all"]: d = dedup(d, P["dedup_iou"], P["dedup_inside"])
        if len(d) and P["min_size_px"] > 0: d = d[(d[:, 1:].view(-1, 4, 2).amax(1) - d[:, 1:].view(-1, 4, 2).amin(1)).amax(-1) >= P["min_size_px"]]
        # 5. corner-peak head: confirm weak gates (>= confirm_n in-image corners with a peak nearby), snap confirmed corners; peaks = gate corners
        if P["confirm"] and len(d):
            c = d[:, 1:].view(-1, 4, 2).clone(); size = (c.amax(1) - c.amin(1)).amax(-1); nconf = torch.zeros(len(d)); hi = P["confirm_hi"]
            if len(pk):
                dd = (c[:, :, None, :] - pk[None, None, :, 2:4]).norm(dim=-1); mn, j = dd.min(-1)
                ok = mn < torch.clamp(P["confirm_r"] * size, min=P["confirm_rmin"])[:, None]
                inimg = (c[..., 0] >= 0) & (c[..., 0] < w) & (c[..., 1] >= 0) & (c[..., 1] < h); ok = ok & inimg; nconf = ok.sum(1).float(); c[ok] = pk[j[ok], 2:4]
            keep = (d[:, 0] >= hi) | (nconf >= P["confirm_n"])
            d = torch.cat((torch.where(d[:, :1] >= hi, d[:, :1], d[:, :1].clamp(min=hi)), c.reshape(-1, 8)), 1)[keep]
        peaks = peaks_from(d) if P["confirm"] else torch.cat((pk, peaks_from(d)))
        t5 = time.perf_counter()
        return dict(gates=d, peaks=peaks, seg=self._seg11(o), depth=o["depth"][0, 0], pooled=o["pooled"][0], seg_in=(o["seg_in"][0].to(torch.uint8) if "seg_in" in o else None), seg_fine=(o["seg_fine"][0].to(torch.uint8) if "seg_fine" in o else None), head=o.get("head"), p_indoor=self.p_indoor,
                    ms=dict(prep=1e3 * (t1 - t0), engine=1e3 * (t2 - t1), heads_post=1e3 * (t3 - t2), refiner=1e3 * (t4 - t3), confirm=1e3 * (t5 - t4), total=1e3 * (t5 - t0)))

    def _post_fast(self, o, img, h, w, t0, t1, t2):
        """Same steps 1-5 as the torch path; device work (peaks, border filter) stays on the GPU, then ONE copy of ~0.3 k floats to numpy."""
        P = self.P; K = o["gates"].shape[1]
        pk5 = self._peaks(o, img, h, w) if self.graphs else gpu_peaks(o["cpk_heat"], o["cpk_off"], img, (h, w), P["cpk_thr"])
        flat = torch.cat((o["gates"][0].float().reshape(-1), o["ng_gates"][0].float().reshape(-1), pk5.reshape(-1))).cpu().numpy()
        G = flat[:K * 9].reshape(K, 9); N2 = flat[K * 9:2 * K * 9].reshape(-1, 9); pk = flat[2 * K * 9:].reshape(-1, 5); pk = pk[pk[:, 4] > 0.5, :4]
        wh = np.array([w, h], np.float32)
        # 1. main gates
        sc = _sig(G[:, 0]); rows = [np.concatenate(([sc[q]], order_np(G[q, 1:].reshape(4, 2) * wh).reshape(-1))) for q in np.nonzero(sc > P["thr"])[0]]
        d = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 9), np.float32)
        # 2. near-gate head merge
        nsc = _sig(N2[:, 0]); keep = nsc >= P["ng_thr"]; c = N2[keep, 1:].reshape(-1, 4, 2) * wh; nsc = nsc[keep]
        if len(c):
            size = (c.max(1) - c.min(1)).max(-1); big = size >= P["ng_min"] * max(h, w); c, nsc, size = c[big], nsc[big], size[big]
        for k in range(len(c)):
            row = np.concatenate(([nsc[k]], c[k].reshape(-1))).astype(np.float32)
            if len(d):
                dc = d[:, 1:].reshape(-1, 4, 2); cands = [np.roll(c[k], r, 0) for r in range(4)] + [np.roll(c[k][::-1], r, 0) for r in range(4)]
                dist = np.stack([np.linalg.norm(dc - q[None], axis=-1).mean(-1) for q in cands]).min(0); j = int(dist.argmin())
                if dist[j] < 0.25 * size[k]:
                    dsz = (dc[j].max(0) - dc[j].min(0)).max()
                    if size[k] >= dsz and size[k] >= P["ng_min"] * max(h, w) and nsc[k] >= P["ng_thr"]:
                        d = d.copy(); d[j] = np.concatenate(([max(d[j, 0], nsc[k])], c[k].reshape(-1)))
                    continue
            d = np.concatenate((d, row[None])) if len(d) else row[None]
        t3 = time.perf_counter()
        # 3. corner refiner (patches sampled on the device)
        if len(d):
            if self.graphs: r = self._refine(img, d[:, 1:])
            else: r = refine_vec(self.be.refiner, img, torch.from_numpy(np.ascontiguousarray(d[:, 1:])).to(self.dev), P["refiner_margin"], P["refiner_vis_thr"]).cpu().numpy()
            d = np.concatenate((d[:, :1], r), 1).astype(np.float32)
        t4 = time.perf_counter()
        # 4. dedup + min size
        if P["dedup_all"]: d = dedup_np(d, P["dedup_iou"], P["dedup_inside"])
        if len(d) and P["min_size_px"] > 0: cc = d[:, 1:].reshape(-1, 4, 2); d = d[(cc.max(1) - cc.min(1)).max(-1) >= P["min_size_px"]]
        # 5. corner-peak confirmation
        if P["confirm"] and len(d):
            c = d[:, 1:].reshape(-1, 4, 2).copy(); size = (c.max(1) - c.min(1)).max(-1); nconf = np.zeros(len(d), np.float32); hi = P["confirm_hi"]
            if len(pk):
                dd = np.linalg.norm(c[:, :, None, :] - pk[None, None, :, 2:4], axis=-1); j = dd.argmin(-1); mn = np.take_along_axis(dd, j[..., None], -1)[..., 0]
                ok = mn < np.maximum(P["confirm_r"] * size, P["confirm_rmin"])[:, None]
                inimg = (c[..., 0] >= 0) & (c[..., 0] < w) & (c[..., 1] >= 0) & (c[..., 1] < h); ok &= inimg; nconf = ok.sum(1).astype(np.float32); c[ok] = pk[j[ok], 2:4]
            keep = (d[:, 0] >= hi) | (nconf >= P["confirm_n"])
            d = np.concatenate((np.maximum(d[:, :1], hi), c.reshape(-1, 8)), 1)[keep].astype(np.float32)
        peaks = peaks_from_np(d) if P["confirm"] else np.concatenate((pk, peaks_from_np(d)))
        t5 = time.perf_counter()
        return dict(gates=torch.from_numpy(d), peaks=torch.from_numpy(np.ascontiguousarray(peaks, dtype=np.float32)), seg=self._seg11(o), depth=o["depth"][0, 0], pooled=o["pooled"][0], seg_in=(o["seg_in"][0].to(torch.uint8) if "seg_in" in o else None), seg_fine=(o["seg_fine"][0].to(torch.uint8) if "seg_fine" in o else None), head=o.get("head"), p_indoor=self.p_indoor,
                    ms=dict(prep=1e3 * (t1 - t0), engine=1e3 * (t2 - t1), heads_post=1e3 * (t3 - t2), refiner=1e3 * (t4 - t3), confirm=1e3 * (t5 - t4), total=1e3 * (t5 - t0)))
