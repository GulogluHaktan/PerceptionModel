"""Cold random-read benchmark for the disk that will hold data/cache (HDD vs NVMe question).
Usage: python scripts/disk_bench.py <dir> [n_files=400] [file_mb=1] [threads=8]
Writes n_files random files of file_mb MB in <dir>/_disk_bench, evicts them from the page cache
(posix_fadvise DONTNEED; works without root for clean pages), then reads them in random order with several
threads, like a shuffled DataLoader reading cached teacher features. Prints MB/s, files/s, and the number of
dataset images/s this would sustain at `files_per_img` files per image (default 5: two features, image, label, corners).
Files are deleted afterwards."""
import json, os, random, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

d = Path(sys.argv[1]) / "_disk_bench"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
mb = int(sys.argv[3]) if len(sys.argv) > 3 else 1
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
files_per_img = 5
d.mkdir(parents=True, exist_ok=True)
blob = os.urandom(mb * 2**20)
paths = [d / f"f{i}.bin" for i in range(n)]
t = time.time()
for p in paths:
    with open(p, "wb") as f:
        f.write(blob); f.flush(); os.fsync(f.fileno())
write_s = time.time() - t


def evict():
    for p in paths:
        fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def read(p):
    with open(p, "rb") as f:
        return len(f.read())


evict(); random.shuffle(paths)
t = time.time()
with ThreadPoolExecutor(threads) as ex:
    total = sum(ex.map(read, paths))
cold_s = time.time() - t
t = time.time()
with ThreadPoolExecutor(threads) as ex:
    sum(ex.map(read, paths))
warm_s = time.time() - t
shutil.rmtree(d)
res = {"dir": str(d.parent), "files": n, "file_mb": mb, "threads": threads, "write_MBps": round(n * mb / write_s, 1),
       "cold_read_MBps": round(total / 2**20 / cold_s, 1), "cold_files_per_s": round(n / cold_s, 1),
       "cold_images_per_s": round(n / cold_s / files_per_img, 1), "warm_read_MBps": round(total / 2**20 / warm_s, 1)}
print(json.dumps(res))
