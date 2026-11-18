"""Pick the most diverse frames to hand-label as the golden set (farthest-point sampling on cached DINO features).
Usage: python scripts/golden_pick.py <images_dir> <cache_dir> <n> <out_dir> [teacher_name=dino] [size=448]
Creates <out_dir>/images (copies), seg/ and depth/ (empty, for YOUR labels) and picked.txt.
Hand-label seg/<id>.png (class idx per src/pfm/model.py CLASSES, 255 = ignore). Never train on this set."""
import shutil, sys
from pathlib import Path
import torch
sys.path.insert(0, "src")
from pfm.cache import FeatureCache
from pfm.data import ImageFolder

img_dir, cache_dir, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
tname = sys.argv[5] if len(sys.argv) > 5 else "dino"
ds = ImageFolder(img_dir, int(sys.argv[6]) if len(sys.argv) > 6 else 448)
cache = FeatureCache(cache_dir)
ids = [str(p.relative_to(ds.root).with_suffix("")).replace("/", "__") for p in ds.files]
feats = torch.nn.functional.normalize(torch.stack([cache.get(tname, i)[1] for i in ids]), dim=1)
picked = [0]
d = 1 - feats @ feats[0]
while len(picked) < min(n, len(ids)):
    j = int(d.argmax()); picked.append(j); d = torch.minimum(d, 1 - feats @ feats[j])
for sub in ("images", "seg", "depth"):
    (out / sub).mkdir(parents=True, exist_ok=True)
for j in picked:
    shutil.copy(ds.files[j], out / "images" / (ids[j] + ds.files[j].suffix))
(out / "picked.txt").write_text("\n".join(ids[j] for j in picked))
print(f"picked {len(picked)} diverse frames -> {out}/images; now label seg/<id>.png by hand")
