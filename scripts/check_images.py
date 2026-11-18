"""Scan an image folder for unreadable / truncated files BEFORE a long run.
Usage: python scripts/check_images.py <images_dir>
Writes <images_dir>/bad_files.txt (relative paths); ImageFolder then skips them automatically.
Also reports id collisions (ImageFolder refuses to start on them)."""
import os, sys
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
sys.path.insert(0, "src")
from pfm.data import EXT


def ok(p):
    try:
        with Image.open(p) as im:
            im.load()
        return None
    except Exception as e:
        return (str(p), type(e).__name__)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    files = [p for p in root.rglob("*") if p.suffix.lower() in EXT]
    with Pool(os.cpu_count()) as pool:
        bad = [r for r in pool.imap_unordered(ok, files, chunksize=64) if r]
    (root / "bad_files.txt").write_text("\n".join(str(Path(p).relative_to(root)) for p, _ in bad))
    print(f"{len(files)} images scanned, {len(bad)} unreadable" + (f" (e.g. {bad[0]})" if bad else ""))
    from pfm.data import ImageFolder
    ImageFolder(root, 8)
    print("no id collisions")
