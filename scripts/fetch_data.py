"""Fetch a licence-tracked public image mix into data/images/<source>/ and log it to data/LICENSES.csv.

Usage: python scripts/fetch_data.py [--n 2000] [--only name1,name2] [--include-nc]
Licences below were read from each Hugging Face dataset card tag on 2026-11-05, NOT legally vetted.
Non-commercial (NC) sources are excluded unless --include-nc. CC-BY-SA is share-alike: whether it
extends to trained weights is unsettled, so decide before shipping a model.
"""
import argparse, csv, io, sys
from pathlib import Path
from PIL import Image

# name -> (hf repo, licence tag, domain, nc?, regex of image files inside the repo)
SOURCES = {
    "airsim_drone": ("asterphys/airsim-drone-dataset", "apache-2.0", "sim drone RGB (+depth npy, masks in repo)", False, r"^train/rgb/.*\.png$"),
    "aeroscapes": ("dronefreak/Aeroscapes", "cc-by-sa-4.0", "drone segmentation", False, r"^data/images/train/.*\.jpg$"),
    "carla_driving": ("jkdxbns/autonomous-driving-carla", "mit", "sim driving", False, r"^yolo_dataset/train/images/.*\.jpg$"),
    "seadronessee": ("dronefreak/SeaDronesSee", "cc0-1.0", "drone sea", False, r"^data/images/train/.*\.jpg$"),
    "uavid": ("dronefreak/UAVid-2020", "cc-by-nc-sa-4.0", "uav segmentation", True, r".*\.(png|jpg)$"),
    # jarvissun/UAVReason_depth (apache-2.0) skipped: depth only, no RGB, 507 GB.
}


def fetch(repo, pattern, out, n, max_side=1024):
    """Evenly sample n files matching `pattern` from a HF dataset repo, resize, save as jpg."""
    import re
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if re.match(pattern, f))
    if not files:
        raise RuntimeError(f"{repo}: pattern matched nothing")
    step = max(1, len(files) // n)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, f in enumerate(files[::step][:n]):
        img = Image.open(hf_hub_download(repo, f, repo_type="dataset")).convert("RGB")
        img.thumbnail((max_side, max_side))
        img.save(out / f"{i:06d}.jpg", quality=92)
        saved += 1
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--only", default="")
    ap.add_argument("--include-nc", action="store_true")
    ap.add_argument("--root", default="data/images")
    a = ap.parse_args()
    ledger = Path("data/LICENSES.csv")
    ledger.parent.mkdir(exist_ok=True)
    new = not ledger.exists()
    names = [x for x in a.only.split(",") if x] or list(SOURCES)
    with open(ledger, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["source", "hf_repo", "license_tag", "domain", "images_saved"])
        for k in names:
            repo, lic, dom, nc, pat = SOURCES[k]
            if nc and not a.include_nc:
                print("skip (non-commercial):", k); continue
            try:
                n = fetch(repo, pat, Path(a.root) / k, a.n)
                print(f"{k}: {n} images ({lic})")
                w.writerow([k, repo, lic, dom, n])
            except Exception as e:
                print(f"{k}: FAILED {type(e).__name__}: {str(e)[:150]}")
