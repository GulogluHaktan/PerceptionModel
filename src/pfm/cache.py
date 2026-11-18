"""Teacher feature cache (D1: cache once, train both students from it).

A teacher is a callable: image batch (B,3,H,W) -> (spatial_map (B,C,h,w), summary (B,C)).
Features are stored as float16 .pt shards keyed by sample id. Disk use is the main cost
(unverified: how much cache we need per image - measure on Monday).
"""
from __future__ import annotations

from pathlib import Path

import torch


class FeatureCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_meta(self, meta: dict):
        import json
        (self.root / "meta.json").write_text(json.dumps(meta))

    def read_meta(self):
        import json
        p = self.root / "meta.json"
        return json.loads(p.read_text()) if p.exists() else None

    def check_size(self, size: int):
        """Fail early (not hours into training) when the cache was built at another resolution."""
        m = self.read_meta()
        if m and m.get("size") != size:
            raise ValueError(f"cache was built at size {m.get('size')} but config says size {size}: "
                             f"rebuild the cache (delete {self.root}) or fix the config")

    def path(self, teacher: str, sample_id: str) -> Path:
        d = self.root / teacher
        d.mkdir(exist_ok=True)
        return d / f"{sample_id}.pt"

    def has(self, teacher: str, sample_id: str) -> bool:
        return self.path(teacher, sample_id).exists()

    def put(self, teacher: str, sample_id: str, fmap: torch.Tensor, summary: torch.Tensor):
        torch.save({"fmap": fmap.half().cpu(), "summary": summary.half().cpu()}, self.path(teacher, sample_id))

    def get(self, teacher: str, sample_id: str):
        d = torch.load(self.path(teacher, sample_id), map_location="cpu")
        return d["fmap"].float(), d["summary"].float()


@torch.no_grad()
def build_cache(cache: FeatureCache, teacher_name: str, teacher_fn, loader, device="cuda"):
    """Run a teacher over a loader yielding (ids, images) and store features."""
    for ids, imgs in loader:
        todo = [i for i, s in enumerate(ids) if not cache.has(teacher_name, s)]
        if not todo:
            continue
        fmap, summ = teacher_fn(imgs[todo].to(device))
        for j, i in enumerate(todo):
            cache.put(teacher_name, ids[i], fmap[j], summ[j])
