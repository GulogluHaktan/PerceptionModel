import pytest
import torch

from pfm import CLASSES, PerceptionModel
from pfm.distill import distill_loss, self_distill_loss
from pfm.gate import GateConfig, decide


@pytest.mark.parametrize("kind", ["vit", "hybrid"])
def test_forward_backward_and_add_teacher(kind):
    m = PerceptionModel(kind, teacher_dims={"dino": 64})
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out["depth"].shape == (2, 1, 224, 224)
    assert out["seg"].shape == (2, len(CLASSES), 224, 224)
    assert out["corners"].shape == (2, 4, 224, 224)
    fake = {"dino": (torch.randn(2, 64, 14, 14), torch.randn(2, 64))}
    loss, _ = distill_loss(out["teacher"], fake)
    loss.backward()
    m.add_teacher("siglip", 32)  # continual: new teacher, same backbone
    assert "siglip" in m(x)["teacher"]
    assert self_distill_loss(out, {k: v.detach() for k, v in out.items() if k != "teacher"}).item() < 1e-3


def test_gate():
    cfg = GateConfig()
    cur = {"seg_miou": 0.5, "depth_absrel": 0.2}
    assert decide(dict(cur), cur, cfg)["recommend_promote"]
    assert not decide({"seg_miou": 0.4, "depth_absrel": 0.2}, cur, cfg)["recommend_promote"]
    assert not decide({"seg_miou": 0.5}, cur, cfg)["recommend_promote"]  # missing metric = reject


def test_label_helpers():
    import numpy as np
    from pfm.labeler import mask_to_corners, order_corners
    from pfm.train import corner_heatmap

    c = order_corners([[10, 10], [0, 10], [10, 0], [0, 0]])
    assert c.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]  # tl tr br bl
    m = np.zeros((100, 100), np.uint8); m[20:60, 30:80] = 1
    q = mask_to_corners(m)
    assert q is not None and abs(q[:, 0].min() - 30) < 2 and abs(q[:, 1].max() - 59) < 2
    hm = corner_heatmap(np.array([[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]], np.float32), 100, 3.0)
    assert hm.shape == (4, 100, 100) and hm[0, 20, 20] > 0.99 and hm[2, 80, 80] > 0.99
    assert corner_heatmap(np.zeros((0, 4, 2), np.float32), 50, 3.0).sum() == 0


def test_energy_parser_and_integration():
    from pfm.energy import EnergyMeter, parse_power_mw

    line = "RAM 3000/7620MB GR3D_FREQ 45% VDD_IN 5123mW/5100mW VDD_CPU_GPU_CV 1234mW/1200mW"
    assert parse_power_mw(line) == 5123
    assert parse_power_mw(line, "VDD_CPU_GPU_CV") == 1234
    assert parse_power_mw("nothing here") is None
    assert abs(EnergyMeter.integrate([(0.0, 5000), (2.0, 5000)]) - 10.0) < 1e-9  # 5 W for 2 s = 10 J


def test_gate_catches_slow_drift_with_an_anchor():
    cfg = GateConfig(floors={}, max_regress=0.02, max_total_regress=0.04)
    best = {"seg_miou": 0.50}; steps = [0.485, 0.47, 0.455]                 # every step loses 0.015: each passes the step rule
    prev = {"seg_miou": 0.50}
    for i, v in enumerate(steps):
        cand = {"seg_miou": v}
        assert decide(cand, prev, cfg)["recommend_promote"]                 # without an anchor the drift is never seen
        anchored = decide(cand, prev, cfg, best)
        assert anchored["recommend_promote"] == (i < 2)                     # 0.455 is 0.045 below the best: rejected
        prev = cand
    assert "drift vs best" in decide({"seg_miou": 0.455}, prev, cfg, best)["reasons"][0]


def test_slot_head_and_targets():
    import torch
    from pfm.train import slot_targets, slot_loss
    m = PerceptionModel("vit")
    out = m(torch.randn(2, 3, 224, 224), with_teacher_heads=False)
    assert out["slots"].shape == (2, 4, 9)
    rows = [{"corners_px": [[-40, 50], [120, 50], [120, 200], [-40, 200]], "depths": [2, 2, 2, 2]},      # partly out of view: kept, coords outside 0..1
            {"corners_px": [[10, 10], [20, 10], [20, 20], [10, 20]], "depths": [30, 30, 30, 30]},          # far, small
            {"corners_px": [[300, 10], [400, 10], [400, 90], [300, 90]], "depths": [3] * 4},                # fully outside: dropped
            {"corners_px": [[10, 10], [20, 10], [20, 20], [10, 20]], "depths": [-1, 2, 2, 2]}]              # behind camera: dropped
    t = slot_targets(rows)
    assert t[:, 0].tolist() == [1, 1, 0, 0] and t[0, 1] < 0 and t[0, 8] > 1   # nearest (largest) first
    assert slot_loss(out["slots"], torch.stack([t, t])).isfinite()


def test_slot_loss_skips_hidden_corners():
    from pfm.train import slot_targets, slot_loss
    rows = [{"corners_px": [[10, 10], [60, 12], None, None]}]
    t = slot_targets(rows)[None]
    assert t[0, 0, 0] == 1 and torch.isnan(t[0, 0, 5:]).all() and not torch.isnan(t[0, 0, 1:5]).any()
    pred = torch.zeros(1, 4, 9, requires_grad=True)
    loss = slot_loss(pred, t); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert slot_targets([{"corners_px": [[10, 10], None, None, None]}])[0, 0] == 0   # one known corner: not a target


def test_dense_gate_head_targets_loss_decode(monkeypatch):
    import numpy as np
    from pfm.train import gdet_targets, gdet_loss
    rows = [{"corners_px": [[100, 60], [160, 60], [160, 120], [100, 120]]}, {"corners_px": [[10, 150], [40, 150], None, None]}]
    heat, off = gdet_targets(rows, 56)
    assert int((heat >= 0.999).sum()) == 2
    ci, cj = (int(v) for v in np.argwhere(heat.numpy() >= 0.999)[0])          # first peak (row order): the big gate, centre (130,90)/(256,192)
    c = off[:, ci, cj].view(4, 2) + torch.tensor([(cj + 0.5) / 56, (ci + 0.5) / 56])
    assert torch.allclose(c[0], torch.tensor([100 / 256, 60 / 192]), atol=1e-5)
    monkeypatch.setenv("PFM_GATE_HEAD", "dense")
    from pfm.model import DenseHeads
    hd = DenseHeads(); raw = torch.full((1, 9, 56, 56), -8.0); raw[0, 0, ci, cj] = 6.0; raw[0, 1:, ci, cj] = off[:, ci, cj]
    dec = hd.decode_gdet(raw)
    assert dec.shape == (1, 4, 9) and dec[0, 0, 0] > 0 and dec[0, 1, 0] < 0
    assert torch.allclose(dec[0, 0, 1:3], torch.tensor([100 / 256, 60 / 192]), atol=1e-5)
    rp = torch.zeros(1, 9, 56, 56, requires_grad=True); l = gdet_loss(rp, heat[None], off[None]); l.backward()
    assert torch.isfinite(l) and torch.isfinite(rp.grad).all()


def test_augmentation_flip_keeps_corner_order(tmp_path):
    import json, random
    import numpy as np
    from PIL import Image
    from pfm.train import CachedDS
    from pfm.data import ImageFolder
    (tmp_path / "img/a").mkdir(parents=True); (tmp_path / "lab/corners_full").mkdir(parents=True)
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp_path / "img/a/x.png")
    json.dump([{"corners_px": [[40, 40], [100, 40], [100, 100], [40, 100]]}], open(tmp_path / "lab/corners_full/a__x.json", "w"))
    class C:
        def get(self, t, s): return torch.zeros(4, 8, 8), torch.zeros(4)
    ds = CachedDS(ImageFolder(tmp_path / "img", 64), C(), [{"name": "dino"}], tmp_path / "lab", aug=True, grid=8)
    random.seed(0)
    for _ in range(20):
        out = ds[0]; sl = out[8]
        if sl[0, 0] > 0:
            c = sl[0, 1:].view(4, 2)
            assert c[0, 0] < c[1, 0] and c[3, 0] < c[2, 0] and c[0, 1] < c[3, 1]      # TL left of TR, BL left of BR, top above bottom


def test_gdet_dedup_keeps_nested_drops_duplicate(monkeypatch):
    """Dense gate decode: two peaks that decode to the same corners collapse into one; a small gate inside a large one survives."""
    import torch
    monkeypatch.setenv("PFM_GATE_HEAD", "dense")
    from pfm.model import DenseHeads
    d = DenseHeads(); g = 28; raw = torch.zeros(1, 9, g, g); raw[:, 0] = -10
    def put(i, j, score, cx, cy, half):
        raw[0, 0, i, j] = score
        c = torch.tensor([cx - half, cy - half, cx + half, cy - half, cx + half, cy + half, cx - half, cy + half])
        raw[0, 1:, i, j] = c - torch.tensor([(j + 0.5) / g, (i + 0.5) / g]).repeat(4)
    put(10, 10, 5.0, 0.4, 0.4, 0.2); put(10, 13, 4.0, 0.41, 0.4, 0.2)      # same large gate from two peaks
    put(12, 11, 3.0, 0.42, 0.44, 0.02)                                      # small gate seen through it
    monkeypatch.setenv("PFM_GDET_DEDUP", "0.25"); s = d.decode_gdet(raw)[0]
    assert (s[:, 0] > 0).sum() == 2 and abs(s[1, 1:].view(4, 2)[:, 0].max() - s[1, 1:].view(4, 2)[:, 0].min() - 0.04) < 1e-4
    monkeypatch.setenv("PFM_GDET_DEDUP", "0"); assert (d.decode_gdet(raw)[0][:, 0] > 0).sum() == 3


def test_weak_gate_seg_and_not_gate_loss():
    """Real gate frames with corner labels only: band just outside the opening = gate, opening and far away = not gate, halo = unknown; the loss falls when P(gate) falls there."""
    import numpy as np, torch
    from pfm.gate_seg import weak_gate_seg, GATE, NOT_GATE, IGNORE
    from pfm.train import not_gate_loss
    g = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], np.float32)
    s = weak_gate_seg([g], 200, 200)
    assert s[100, 100] == NOT_GATE and s[5, 5] == NOT_GATE and s[100, 78] == GATE and s[100, 60] == IGNORE
    part = g.copy(); part[0] = np.nan
    assert (weak_gate_seg([part], 200, 200) == GATE).sum() == 0 and weak_gate_seg([], 50, 50).min() == NOT_GATE
    logits = torch.zeros(1, 11, 4, 4); mask = torch.ones(1, 4, 4, dtype=torch.bool); hi = logits.clone(); hi[:, 1] = 5.0; lo = logits.clone(); lo[:, 1] = -5.0
    assert not_gate_loss(hi, mask) > not_gate_loss(logits, mask) > not_gate_loss(lo, mask)


def test_absrel_both_matches_space():
    """A prediction that is exactly inverse depth scores ~0 under the inverse fit and clearly worse under the linear fit; a metric prediction the other way round."""
    import numpy as np
    from pfm.evaluate import absrel_both
    g = np.linspace(1, 20, 400).reshape(20, 20); v = np.ones_like(g, bool)
    lin, inv = absrel_both(3.0 / g + 0.2, g, v); assert inv < 1e-6 < 0.05 < lin
    lin, inv = absrel_both(2.0 * g + 1.0, g, v); assert lin < 1e-6 < 0.05 < inv


def test_half_gate_crop_and_bar_rule():
    import random
    import numpy as np
    from pfm.train import half_gate_crop, gate_list, frame_in_view
    c = np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]], np.float32); rng = random.Random(0)
    for n_in in (0, 1):
        for _ in range(20):
            hc = half_gate_crop(c, n_in, rng)
            assert hc is not None
            s, x0, y0 = hc; q = (c - [x0, y0]) * s
            assert 0 <= x0 <= 1 - 1 / s and 0 <= y0 <= 1 - 1 / s
            assert int((((q >= 0) & (q < 1)).all(1)).sum()) == n_in and frame_in_view(q, 1.0, 1.0) >= 0.3
    # a bar crossing the image with all corners outside counts as a gate; the same gate far outside the image does not; a gate seen from inside its opening does not
    bar = [{"corners_px": [[-100, 96], [100, -50], [300, 96], [100, 400]], "depths": [2] * 4}]
    away = [{"corners_px": [[300, 10], [400, 10], [400, 100], [300, 100]], "depths": [2] * 4}]
    inside = [{"corners_px": [[-50, -50], [300, -50], [300, 250], [-50, 250]], "depths": [2] * 4}]
    assert len(gate_list(bar)) == 1 and len(gate_list(away)) == 0 and len(gate_list(inside)) == 0


def test_complete_square_and_centre():
    import numpy as np
    from pfm.train import complete_square, gdet_targets
    full = np.array([[0.25, 0.25], [0.5, 0.25], [0.5, 0.5833], [0.25, 0.5833]], np.float32)     # a 64 px square at 256x192
    c3 = full.copy(); c3[2] = np.nan
    assert np.allclose(complete_square(c3)[2], full[2], atol=1e-3)
    for i in range(4):      # any single visible side rebuilds the square
        c2 = full.copy(); c2[(i + 2) % 4] = np.nan; c2[(i + 3) % 4] = np.nan
        assert np.allclose(complete_square(c2), full, atol=2e-3)
    rows = [{"corners_px": [[64, 48], [128, 48], None, None], "depths": [5] * 4}]
    h0, _ = gdet_targets(rows, 28); h1, _ = gdet_targets(rows, 28, complete=True)
    p0 = np.unravel_index(h0.argmax(), h0.shape); p1 = np.unravel_index(h1.argmax(), h1.shape)
    assert p0[0] == int(48 / 192 * 28) and p1[0] == int(80 / 192 * 28) and p0[1] == p1[1]     # centre moves from the visible side to the middle of the square


def test_zoom_out_shrinks_gate_and_keeps_it_inside(tmp_path):
    import json, random
    import numpy as np
    from PIL import Image
    from pfm.train import CachedDS
    from pfm.data import ImageFolder
    (tmp_path / "img/a").mkdir(parents=True); (tmp_path / "lab/corners_full").mkdir(parents=True)
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp_path / "img/a/x.png")
    json.dump([{"corners_px": [[40, 40], [100, 40], [100, 100], [40, 100]]}], open(tmp_path / "lab/corners_full/a__x.json", "w"))
    class C:
        def get(self, t, s): return torch.zeros(4, 8, 8), torch.zeros(4)
    ds = CachedDS(ImageFolder(tmp_path / "img", 64), C(), [{"name": "dino"}], tmp_path / "lab", aug=True, grid=8); ds.zoom_out = 1.0
    random.seed(1); w0 = 60 / 256
    for _ in range(20):
        out = ds[0]; c = out[8][0, 1:].view(4, 2)
        assert out[8][0, 0] > 0 and (c >= 0).all() and (c <= 1).all()
        w = float(c[:, 0].max() - c[:, 0].min()); assert 0.5 * w0 - 1e-4 <= w <= w0 + 1e-4
        assert out[0].shape == (3, 64, 64) and out[10].max() > 0.9


def test_close_gate_indices(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    from pfm.train import close_gate_indices
    from pfm.data import ImageFolder
    for d in ("img/real_a", "img/sim_a", "lab/corners_full"): (tmp_path / d).mkdir(parents=True)
    for n in ("real_a/near", "real_a/far", "real_a/edge", "sim_a/near"): Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp_path / f"img/{n}.png")
    big = [[20, 20], [180, 20], [180, 170], [20, 170]]; small = [[100, 100], [120, 100], [120, 120], [100, 120]]; edge = [[-80, 20], [60, 20], [60, 190], [-80, 300]]
    for n, c in (("real_a__near", big), ("real_a__far", small), ("real_a__edge", edge), ("sim_a__near", big)): json.dump([{"corners_px": c}], open(tmp_path / f"lab/corners_full/{n}.json", "w"))
    im = ImageFolder(tmp_path / "img", 64); got = {im.sid(im.files[i]) for i in close_gate_indices(im, tmp_path / "lab")}
    assert got == {"real_a__near", "real_a__edge"}      # the edge gate has two corners in view, 170 px apart; sim frames are not boosted


def test_gdet_size_levels_keep_nested_gates(monkeypatch):
    """Two gates with the same centre (one seen through the other): with size levels both get a target and both are decoded; without levels one cell holds only one."""
    import torch
    from pfm.train import gdet_targets, gdet_loss
    sq = lambda half: [[128 - half, 96 - half], [128 + half, 96 - half], [128 + half, 96 + half], [128 - half, 96 + half]]
    rows = [{"corners_px": sq(60)}, {"corners_px": sq(8)}]
    heat, off = gdet_targets(rows, 56, levels=(24.0, 80.0))
    assert heat.shape == (3, 56, 56) and off.shape == (24, 56, 56)
    assert heat[0].max() == 1 and heat[2].max() == 1 and heat[1].max() == 0      # 16 px gate on level 0, 120 px gate on level 2
    h1, o1 = gdet_targets(rows, 56); assert h1.shape == (56, 56) and o1.shape == (8, 56, 56)      # no levels: old format
    hb, _ = gdet_targets([{"corners_px": sq(11)}], 56, levels=(24.0, 80.0)); assert hb[0].max() == 1 and hb[1].max() == 1      # 22 px: within 20% of the threshold, on both levels
    monkeypatch.setenv("PFM_GATE_HEAD", "dense"); monkeypatch.setenv("PFM_GDET_LEVELS", "24,80")
    from pfm.model import DenseHeads
    d = DenseHeads(); assert d.gdet_levels == 3
    raw = torch.full((1, 27, 56, 56), -10.0); raw[:, 1:9] = 0; raw[:, 10:18] = 0; raw[:, 19:27] = 0
    i, j = (heat[0] == 1).nonzero()[0].tolist()
    for lv in (0, 2): raw[0, 9 * lv, i, j] = 5.0; raw[0, 9 * lv + 1:9 * lv + 9, i, j] = off[8 * lv:8 * lv + 8, i, j]
    s = d.decode_gdet(raw)[0]; got = sorted(float(s[q, 1:].view(4, 2)[:, 0].max() - s[q, 1:].view(4, 2)[:, 0].min()) for q in range(s.shape[0]) if s[q, 0] > 0)
    assert len(got) == 2 and abs(got[0] - 16 / 256) < 1e-4 and abs(got[1] - 120 / 256) < 1e-4
    rp = torch.zeros(2, 27, 56, 56, requires_grad=True); l = gdet_loss(rp, torch.stack([heat, heat]), torch.stack([off, off])); l.backward(); assert torch.isfinite(l)


def test_side_view_warp_moves_corners_with_the_picture(tmp_path):
    """Side-view augmentation: a bright dot painted on one gate corner must land where the warped corner label says (also when flipped), and the gate must get narrower relative to its height."""
    import json, random
    import numpy as np
    from PIL import Image
    from pfm.train import CachedDS
    from pfm.data import ImageFolder
    (tmp_path / "img/a").mkdir(parents=True); (tmp_path / "lab/corners_full").mkdir(parents=True)
    a = np.zeros((192, 256, 3), np.uint8); a[58:63, 98:103] = 255; Image.fromarray(a).save(tmp_path / "img/a/x.png")      # dot at the top-left corner (100, 60)
    json.dump([{"corners_px": [[100, 60], [160, 60], [160, 120], [100, 120]]}], open(tmp_path / "lab/corners_full/a__x.json", "w"))
    class C:
        def get(self, t, s): return torch.zeros(4, 8, 8), torch.zeros(4)
    ds = CachedDS(ImageFolder(tmp_path / "img", 128), C(), [{"name": "dino"}], tmp_path / "lab", aug=False, grid=16); ds.aug = True; ds.side_view = 1.0
    random.seed(3); torch.manual_seed(0); narrow = 0; n = 0
    for _ in range(40):
        out = ds[0]; x, sl = out[0], out[8]
        if sl[0, 0] <= 0 or x.max() < 0.5: continue
        c = sl[0, 1:].view(4, 2).numpy(); yy, xx = np.unravel_index(int(x.sum(0).argmax()), (128, 128)); dot = np.array([(xx + 0.5) / 128, (yy + 0.5) / 128])
        assert np.abs(c - dot).sum(1).min() < 0.06, (c, dot)      # one labelled corner sits on the dot
        w = np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[3] - c[2]); h = np.linalg.norm(c[0] - c[3]) + np.linalg.norm(c[1] - c[2]); n += 1; narrow += (w * 256) < (h * 192) * 1.001
        assert c[0, 0] < c[1, 0] and c[0, 1] < c[3, 1]
    assert n >= 10 and narrow >= 0.8 * n      # the squeeze factor may be 1 (keystone only)
