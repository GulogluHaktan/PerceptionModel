"""bigE = bigD recipe (stride-4 dense gate head, augmentation, gate-free negatives, class weights, 10 epochs) + BDD100K segmentation frames (7,000 real dashcam frames, also gate-free negatives) + corrected ADE20K labels (labels_E).
Generic: tag, student kind and any config key can be overridden, so the same chain trains the hybrid (FastViT) student and ablations on the bigE image set.
Usage: [KIND=vit|hybrid] [WAIT=~/x.log (start after that log has CHAIN_END)] python scripts/bige_chain.py [tag=bigE] [extra yaml 'k: v' ...]
Evaluations use the decoded dense head (PFM_GATE_HEAD/PFM_GDET_STRIDE from the config) with duplicate removal (default in decode_gdet)."""
import os, subprocess, sys, time
from pathlib import Path
import yaml
tag = sys.argv[1] if len(sys.argv) > 1 else "bigE"; KIND = os.environ.get("KIND", "vit"); PY = ".venv/bin/python"; M = Path("data/mix")
def sh(cmd, check=True):
    print(">>", cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode and check: print("FAILED", cmd, flush=True); sys.exit(1)
def relink(f, d):      # link d -> f; untouched when already right, replaced in one step otherwise (two chains may prepare the same label folder while another run reads it)
    if d.is_symlink() and os.readlink(d) == str(f.resolve()): return
    t = d.with_name(f".{d.name}.{os.getpid()}"); os.symlink(f.resolve(), t); os.replace(t, d)
if os.environ.get("WAIT"):
    while "CHAIN_END" not in Path(os.path.expanduser(os.environ["WAIT"])).read_text(): time.sleep(30)
n_bdd = len(list(Path("data/general/train/images/bdd").glob("*.jpg"))); assert n_bdd >= 7000, f"BDD not prepared ({n_bdd} frames): run scripts/bdd_prepare.py"
img = M / "bigE/images"; lock = M / "bigE.lock"
while lock.exists(): time.sleep(20)        # another chain is building the shared image set / cache
if not (M / "bigE/ready").exists():
    lock.touch()
    try:
        if len(list((M / "cache/dino").glob("bdd__*.pt"))) < n_bdd or len(list((M / "cache/depth").glob("bdd__*.pt"))) < n_bdd:
            base = yaml.safe_load(open("configs/sim_v2_5090.yaml")); (M / "cache_bdd_src/bdd").parent.mkdir(exist_ok=True)
            if (M / "cache_bdd_src/bdd").is_symlink(): (M / "cache_bdd_src/bdd").unlink()       # a linked folder is not walked by ImageFolder (rglob): link the files instead
            if not (M / "cache_bdd_src/bdd").exists(): sh(f"mkdir -p {M}/cache_bdd_src/bdd && cp -as $(pwd)/data/general/train/images/bdd/. {M}/cache_bdd_src/bdd/")
            yaml.safe_dump(dict(base, images=str(M / "cache_bdd_src"), cache=str(M / "cache"), labels="data/general/train/labels", cache_batch=32), open(M / "cache_bdd.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_bdd.yaml")
        L = M / "labels_E"       # own label set: shared files linked, ADE labels from the corrected mapping (scripts/ade_relabel_v2.py), BDD labels added
        if not (L / "seg").exists():
            L.mkdir(exist_ok=True)
            for sub in ("corners", "corners_full", "depth"): os.symlink((M / "labels" / sub).resolve(), L / sub)
            sh(f"cp -a {M}/labels/seg {L}/seg")
        n_ade = 0
        for f in Path("data/general/train/labels_v2/seg").glob("ade__*"):
            d = L / "seg" / f.name; relink(f, d); n_ade += 1
        assert n_ade >= 3000, "ADE v2 labels missing: run scripts/ade_relabel_v2.py"
        n_w = 0     # weak gate / not-gate labels for the real gate frames (scripts/make_real_gate_seg.py)
        for f in Path("data/general/train/labels_v2/seg").glob("real_*"):
            d = L / "seg" / f.name; relink(f, d); n_w += 1
        assert n_w >= 6000, "weak gate seg labels missing: run scripts/make_real_gate_seg.py"
        for f in Path("data/general/train/labels/seg").glob("bdd__*"):
            d = L / "seg" / f.name
            if not d.exists() and not d.is_symlink(): os.symlink(f.resolve(), d)
        if not img.exists():
            sh(f"mkdir -p {img} && cp -as $(pwd)/data/mix/bigD/images/. {img}/ && mkdir -p {img}/bdd && cp -as $(pwd)/data/general/train/images/bdd/. {img}/bdd/")
        (M / "bigE/ready").touch()
    finally:
        lock.unlink()
if os.environ.get("ADD_TUS"):      # bigF image set = bigE + TuSimple lane frames (real lane_marking labels, scripts/tusimple_prepare.py); labels linked into labels_E (unused by bigE itself)
    imgF = M / "bigF/images"; n_t = len(list(Path("data/general/train/images/tus").glob("*.jpg"))); assert n_t >= 200, "run scripts/tusimple_prepare.py"
    if len(list((M / "cache/depth").glob("tus__*.pt"))) < n_t:
        base = yaml.safe_load(open("configs/sim_v2_5090.yaml")); sh(f"mkdir -p {M}/cache_tus_src/tus && cp -as $(pwd)/data/general/train/images/tus/. {M}/cache_tus_src/tus/")
        yaml.safe_dump(dict(base, images=str(M / "cache_tus_src"), cache=str(M / "cache"), labels="data/general/train/labels", cache_batch=32), open(M / "cache_tus.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_tus.yaml")
    for f in Path("data/general/train/labels_v2/seg").glob("tus__*"):
        d = M / "labels_E/seg" / f.name
        if not d.exists() and not d.is_symlink(): os.symlink(f.resolve(), d)
    if not imgF.exists(): sh(f"mkdir -p {imgF} && cp -as $(pwd)/{img}/. {imgF}/ && mkdir -p {imgF}/tus && cp -as $(pwd)/data/general/train/images/tus/. {imgF}/tus/")
    REP = int(os.environ.get("TUS_REPEAT", "8"))       # only 267 real lane frames against 52k others: repeat them (augmentation makes the repeats differ); repeats are links (image, cache, label)
    for k in range(1, REP):
        if (imgF / f"tusr{k}").exists(): continue
        sh(f"mkdir -p {imgF}/tusr{k} && cp -as $(pwd)/data/general/train/images/tus/. {imgF}/tusr{k}/")
        for f in Path("data/general/train/images/tus").glob("*.jpg"):
            for sub in ("cache/dino", "cache/depth"):
                d = M / sub / f"tusr{k}__{f.stem}.pt"
                if not d.exists(): os.symlink((M / sub / f"tus__{f.stem}.pt").resolve(), d)
            d = M / "labels_E/seg" / f"tusr{k}__{f.stem}.png"
            if not d.exists(): os.symlink(Path(f"data/general/train/labels_v2/seg/tus__{f.stem}.png").resolve(), d)
    img = imgF
if os.environ.get("ADD_ENV"):      # bigG image set = current set + UAVid train tiles (uav) + RUGD training sequences (rugd), scripts/env_train_prepare.py; their test sets stay untouched
    imgG = M / "bigG/images"; base = yaml.safe_load(open("configs/sim_v2_5090.yaml"))
    for src in ("uav", "rugd"):
        n_s = len(list(Path(f"data/general/train/images/{src}").glob("*.jpg"))); assert n_s >= 500, f"run scripts/env_train_prepare.py ({src}: {n_s})"
        if len(list((M / "cache/depth").glob(f"{src}__*.pt"))) < n_s:
            sh(f"mkdir -p {M}/cache_{src}_src/{src} && cp -as $(pwd)/data/general/train/images/{src}/. {M}/cache_{src}_src/{src}/")
            yaml.safe_dump(dict(base, images=str(M / f"cache_{src}_src"), cache=str(M / "cache"), labels="data/general/train/labels", cache_batch=32), open(M / f"cache_{src}.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_{src}.yaml")
        for f in Path("data/general/train/labels_v2/seg").glob(f"{src}__*"):
            d = M / "labels_E/seg" / f.name
            if not d.exists() and not d.is_symlink(): os.symlink(f.resolve(), d)
    if not imgG.exists():
        sh(f"mkdir -p {imgG} && cp -as $(pwd)/{img}/. {imgG}/")
        for src in ("uav", "rugd"): sh(f"mkdir -p {imgG}/{src} && cp -as $(pwd)/data/general/train/images/{src}/. {imgG}/{src}/")
    img = imgG
LAB = M / "labels_E"
if os.environ.get("LABELS_V3"):     # label set v3 (scripts/sky_labels_v3.py): sky = class 0, BDD vegetation = tree_pole; own label folder labels_H so that labels_E runs stay reproducible
    LAB = M / "labels_H"
    if not (LAB / "seg").exists():
        LAB.mkdir(exist_ok=True)
        for sub in ("corners", "corners_full", "depth"): os.symlink((M / "labels" / sub).resolve(), LAB / sub)
        sh(f"cp -a {M}/labels_E/seg {LAB}/seg")
    n3 = 0
    for f in Path("data/general/train/labels_v3/seg").glob("*.png"):
        d = LAB / "seg" / f.name; relink(f, d); n3 += 1
    assert n3 >= 10000, "run scripts/sky_labels_v3.py and SKY=1 scripts/ade_relabel_v2.py"
if os.environ.get("LABELS_V4"):     # label set v4 (scripts/lane_labels_v4.py): v3 + weak road-paint labels (lane_marking) on the BDD training frames; own folder labels_I
    assert os.environ.get("LABELS_V3"), "LABELS_V4 builds on LABELS_V3"
    LAB4 = M / "labels_I"
    if not (LAB4 / "seg").exists():
        LAB4.mkdir(exist_ok=True)
        for sub in ("corners", "corners_full", "depth"): os.symlink((M / "labels" / sub).resolve(), LAB4 / sub)
        sh(f"cp -a {LAB}/seg {LAB4}/seg")
    n4 = 0
    for f in Path("data/general/train/labels_v4/seg").glob("*.png"):
        d = LAB4 / "seg" / f.name; relink(f, d); n4 += 1
    assert n4 >= 7000, "run scripts/lane_labels_v4.py"
    LAB = LAB4
if os.environ.get("LABELS_V5"):     # label set v5 (scripts/two_wheel_labels_v5.py + SKY=1 TWO=1 ade_relabel_v2.py): v4 + motorcycles / bicycles as class car; own folder labels_J
    assert os.environ.get("LABELS_V4"), "LABELS_V5 builds on LABELS_V4"
    LAB5 = M / "labels_J"
    if not (LAB5 / "seg").exists():
        LAB5.mkdir(exist_ok=True)
        for sub in ("corners", "corners_full", "depth"): os.symlink((M / "labels" / sub).resolve(), LAB5 / sub)
        sh(f"cp -a {LAB}/seg {LAB5}/seg")
    n5 = 0
    for f in Path("data/general/train/labels_v5/seg").glob("*.png"):
        d = LAB5 / "seg" / f.name; relink(f, d); n5 += 1
    assert n5 >= 10000, "run scripts/two_wheel_labels_v5.py and SKY=1 TWO=1 scripts/ade_relabel_v2.py"
    LAB = LAB5
cfg = yaml.safe_load(open(M / "bigD.yaml")); cfg.update(images=str(img), labels=str(LAB), out_dir=f"runs/{tag}", student=KIND, gate_free_prefixes=["diode__", "sun__", "ade__", "aero__", "bdd__", "tus", "uav__", "rugd__"], workers=12)
for kv in sys.argv[2:]: cfg.update(yaml.safe_load(kv))
cp = M / f"{tag}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
if not Path(f"runs/{tag}/{KIND}/final.pt").exists(): sh(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} scripts/train.py {cp}")
E_ = f"PFM_GDET_K={os.environ.get('PFM_GDET_K', '8')} PFM_GDET_LEVELS={','.join(str(t) for t in cfg.get('gdet_levels', ()))} PFM_GATE_HEAD={cfg.get('gate_head', '')} PFM_GDET_STRIDE={cfg.get('gdet_stride', 8)} KINDS={KIND} KIND={KIND} CKPT=runs/{tag}"; RED = "GOLDEN=data/real_mrg/golden FLIGHTS=red_s1_f1,red_s1_f2,red_s1_f3"
sh(f"{E_} {RED} {PY} scripts/eval_real_golden.py")
sh(f"{E_} {PY} scripts/eval_half_gates.py")   # gates with 0-1 corners in view, crops of the unseen place
sh(f"{E_} STEP=2 {PY} scripts/eval_video_consistency.py", check=False)   # real flight video (TII, hall seen): temporal behaviour and recall by gate size
sh(f"{E_} THR=0.3 {RED} {PY} scripts/eval_real_golden.py")
sh(f"{E_} {PY} scripts/eval_real_golden.py")
sh(f"{E_} {PY} scripts/eval_gates_sim.py sim_racec")
sh(f"{E_} {PY} scripts/eval_gates_sim.py sim_racep", check=False)
sh(f"{E_} {PY} scripts/diag_gates_size.py sim_racec", check=False)
sh(f"{E_} EVAL_ROOT=data/heldout TAG={tag} {PY} scripts/eval_students_sim.py sim_indoor", check=False)
sh(f"{E_} {PY} scripts/eval_depth_diode.py")
sh(f"{E_} EVDIR=data/general/eval_kitti {PY} scripts/eval_depth_dir.py", check=False)
for ev in ("sun", "ade", "ade2", "aero", "bdd", "uavid", "uavid_tiles", "rugd", "rugd_heldout", "iddaw", "bdd3", "ade3", "rugd_heldout3", "iddaw3", "bdd5", "ade5", "iddaw5"): sh(f"{E_} EVDIR=data/general/eval_{ev} {PY} scripts/eval_seg_sun.py", check=False)
sh(f"{E_} {PY} scripts/eval_gate_seg_real.py", check=False)
sh(f"{E_} {PY} scripts/eval_lane_tusimple.py", check=False)
sh(f"{E_} {PY} scripts/eval_lane_proxy_bdd.py", check=False)
sh(f"{E_} {PY} scripts/eval_real_negatives.py runs/{tag}", check=False)
sh(f"{E_} EVAL_ROOT=data/heldout TAG={tag} {PY} scripts/eval_students_sim.py sim_outdoor", check=False)
sh(f"{E_} OUT_TAG=_{tag} {PY} scripts/eval_robustness_sim.py 4 | tee ~/{tag}_robust.txt", check=False)
print("CHAIN DONE", flush=True)
