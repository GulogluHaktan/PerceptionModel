"""Datasets. Images are returned in [0,1], fixed square size; ids are stable file stems."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class ImageFolder(Dataset):
    """Recursively reads images. id = relative path with '/' -> '__' and no suffix."""

    def __init__(self, root, size=224):
        self.root, self.size = Path(root), size
        bad = set()
        bf = self.root / "bad_files.txt"      # written by scripts/check_images.py
        if bf.exists():
            bad = set(bf.read_text().splitlines())
        self.files = sorted(p for p in self.root.rglob("*")
                            if p.suffix.lower() in EXT and str(p.relative_to(self.root)) not in bad)
        if not self.files:
            raise FileNotFoundError(f"no images under {root}")
        ids = [self.sid(p) for p in self.files]
        dup = sorted({i for i in ids if ids.count(i) > 1}) if len(ids) < 5000 else self._dups(ids)
        if dup:
            raise ValueError(f"{len(dup)} colliding image ids (same stem with another extension, or a/b vs a__b), "
                             f"e.g. {dup[:3]}. Rename the files: ids would share one cache/label entry.")

    @staticmethod
    def _dups(ids):
        seen, dup = set(), set()
        for i in ids:
            (dup if i in seen else seen).add(i)
        return sorted(dup)

    def sid(self, p):
        return str(p.relative_to(self.root).with_suffix("")).replace("/", "__")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        return self.sid(p), x


class GoldenSet(Dataset):
    """Frozen hand-labelled set. Layout: images/<id>.jpg, seg/<id>.png (class idx),
    depth/<id>.npy (optional, metric or relative). Never trained on (D6/D8)."""

    def __init__(self, root, size=224):
        self.root, self.size = Path(root), size
        self.ids = sorted(p.stem for p in (self.root / "images").glob("*") if p.suffix.lower() in EXT)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        img_p = next((self.root / "images").glob(sid + ".*"))
        x = torch.from_numpy(np.asarray(Image.open(img_p).convert("RGB").resize((self.size,) * 2, Image.BICUBIC)).copy())
        x = x.permute(2, 0, 1).float() / 255
        seg_p = self.root / "seg" / f"{sid}.png"
        seg = torch.from_numpy(np.asarray(Image.open(seg_p).resize((self.size,) * 2, Image.NEAREST)).copy()).long() \
            if seg_p.exists() else torch.full((self.size, self.size), -1)
        dp = self.root / "depth" / f"{sid}.npy"
        depth = torch.from_numpy(np.load(dp)).float()[None, None] if dp.exists() else None
        depth = torch.nn.functional.interpolate(depth, size=(self.size,) * 2, mode="bilinear")[0] \
            if depth is not None else torch.zeros(1, self.size, self.size)
        return x, seg, depth


def make_fake_data(root, n=64, size=224, seed=0, golden=False):
    """Random images (+ labels for a golden set) so the whole pipeline can be tested without data."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    if golden:
        (root / "seg").mkdir(exist_ok=True)
        (root / "depth").mkdir(exist_ok=True)
    for i in range(n):
        base = rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8)
        img = Image.fromarray(base).resize((size, size), Image.BICUBIC)
        img.save(root / "images" / f"{i:05d}.jpg")
        if golden:
            seg = np.kron(rng.integers(0, 11, (4, 4)), np.ones((size // 4, size // 4))).astype(np.uint8)
            Image.fromarray(seg).save(root / "seg" / f"{i:05d}.png")
            np.save(root / "depth" / f"{i:05d}.npy", (np.linspace(1, 10, size)[None] * np.ones((size, 1))).astype(np.float32))
