"""Run each configured teacher once over the image folder and store features."""
import sys, yaml, torch
sys.path.insert(0, "src")
from torch.utils.data import DataLoader
from pfm.cache import FeatureCache, build_cache
from pfm.data import ImageFolder, normalize
from pfm.teachers import build_teacher

cfg = yaml.safe_load(open(sys.argv[1]))
dev = "cuda" if torch.cuda.is_available() else "cpu"
ds = ImageFolder(cfg["images"], cfg["size"])
cache = FeatureCache(cfg["cache"])
cache.check_size(cfg["size"])
dl = DataLoader(ds, batch_size=cfg.get("cache_batch", 64), num_workers=cfg.get("workers", 4))
for t in cfg["teachers"]:
    print("caching", t["name"], t["kind"], flush=True)
    teacher = build_teacher(t["kind"], dev)
    build_cache(cache, t["name"], lambda x: teacher(normalize(x)), dl, dev)
    del teacher; torch.cuda.empty_cache()

cache.write_meta({"size": cfg["size"], "teachers": {t["name"]: t["kind"] for t in cfg["teachers"]}})
