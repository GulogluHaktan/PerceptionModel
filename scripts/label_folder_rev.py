"""Second labelling worker for scripts/label_folder.py: same labeller, same output, but walks the images BACKWARDS. Both workers skip images that already have a label, so two workers
(one forward, one reverse) share the work and finish when they meet; at most one image near the meeting point is labelled twice (identical result).
Usage: python scripts/label_folder_rev.py <images_dir> <labels_out_dir> [size]"""
import sys
sys.path.insert(0, "src")
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pfm.data import ImageFolder
from pfm.labeler import Labeler
root, out, size = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 448
ds = ImageFolder(root, size); (out / "seg").mkdir(parents=True, exist_ok=True); (out / "corners").mkdir(exist_ok=True); lab = Labeler("cuda")
for i in tqdm(range(len(ds) - 1, -1, -1)):
    sid, x = ds[i]
    if (out / "seg" / f"{sid}.png").exists():
        continue
    seg, corners = lab.label(Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
    Image.fromarray(seg).save(out / "seg" / f"{sid}.png"); np.save(out / "corners" / f"{sid}.npy", corners)
