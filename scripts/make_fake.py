import sys
sys.path.insert(0, "src")
from pfm.data import make_fake_data
make_fake_data("data/fake/images_root", n=64)
make_fake_data("data/fake/golden", n=16, seed=1, golden=True)
