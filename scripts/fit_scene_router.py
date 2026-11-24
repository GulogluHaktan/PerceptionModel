"""Indoor / outdoor scene router for module switching on the vehicle (indoor frames -> indoor 22-class module, outdoor -> seg module v2).
Feature = the core's pooled fused stride-8 feature (system output "pooled", 128-d; free at run time). Logistic regression (numpy, standardised).
Indoor: SUN RGB-D train, COCO val indoor (labels_in share >= 0.4 = the coco train part + eval_coco_in), real phone office frames (data/real_phone, test only).
Outdoor: BDD train, IDD-AW eval, RUGD train, UAV, COCO val outdoor (share < 0.1).  Split by source file order: 80 % fit / 20 % test.
Writes runs/modules/scene_router.json {mean, std, w, b, test_acc, ...} (copied into bundle.json by deploy/export_system.py).
Run: env -u PYTHONPATH .venv/bin/python scripts/fit_scene_router.py"""
import os, sys, json, glob, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import train_seg_module as T

N = int(os.environ.get("N", "1500"))


def lists():
    R = random.Random(0); G = lambda pat: sorted(glob.glob(pat))
    coco = []
    for lp in G("data/general/train/labels_in/seg/coco__*.png"):
        a = np.asarray(Image.open(lp)); v = a != 255; sh = float(((a > 0) & v).sum() / max(1, v.sum())); coco.append((lp.replace("labels_in/seg/coco__", "images/coco/").replace(".png", ".jpg"), sh))
    indoor = G("data/general/train/images/sun/*") + [p for p, sh in coco if sh >= 0.4] + G("data/general/eval_coco_in/images/*")
    outdoor = G("data/general/train/images/bdd/*") + G("data/general/eval_iddaw/images/*") + G("data/general/train/images/rugd/*") + G("data/general/train/images/uav/*") + [p for p, sh in coco if sh < 0.1]
    R.shuffle(indoor); R.shuffle(outdoor); return indoor[:N], outdoor[:N], G("data/real_phone/*.jpg")


@torch.no_grad()
def feats(core, files):
    out = []
    for i in range(0, len(files), 16):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((448, 448), Image.BILINEAR)).copy()).permute(2, 0, 1).float() / 255 for f in files[i:i + 16]]).cuda()
        f, _ = T.core_pass(core, x); out.append(f.mean((2, 3)).cpu().numpy())
    return np.concatenate(out)


def main():
    core = T.PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.cuda().eval()
    ind, outd, phone = lists(); print("indoor", len(ind), "outdoor", len(outd), "phone", len(phone), flush=True)
    Xi, Xo, Xp = feats(core, ind), feats(core, outd), feats(core, phone)
    ni, no = int(0.8 * len(Xi)), int(0.8 * len(Xo))
    Xtr = np.concatenate([Xi[:ni], Xo[:no]]); ytr = np.r_[np.ones(ni), np.zeros(no)]; Xte = np.concatenate([Xi[ni:], Xo[no:]]); yte = np.r_[np.ones(len(Xi) - ni), np.zeros(len(Xo) - no)]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = lambda X: (X - mu) / sd
    w = np.zeros(Xtr.shape[1]); b = 0.0; lr = 0.1; lam = 1e-3
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(Z(Xtr) @ w + b))); g = p - ytr; w -= lr * (Z(Xtr).T @ g / len(g) + lam * w); b -= lr * g.mean()
    pr = lambda X: 1 / (1 + np.exp(-(Z(X) @ w + b)))
    acc = float(((pr(Xte) > 0.5) == yte).mean()); acc_in = float((pr(Xi[ni:]) > 0.5).mean()); acc_out = float((pr(Xo[no:]) <= 0.5).mean()); phone_in = float((pr(Xp) > 0.5).mean())
    res = dict(mean=mu.round(6).tolist(), std=sd.round(6).tolist(), w=w.round(6).tolist(), b=round(float(b), 6), test_acc=round(acc, 4), test_indoor_recall=round(acc_in, 4),
               test_outdoor_recall=round(acc_out, 4), phone_office_frames_indoor=round(phone_in, 4), n_fit=int(len(ytr)), n_test=int(len(yte)))
    Path("runs/modules/scene_router.json").write_text(json.dumps(res)); print(json.dumps({k: v for k, v in res.items() if k not in ("mean", "std", "w")}))


if __name__ == "__main__":
    main()
