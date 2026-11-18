import numpy as np
import pytest
from PIL import Image

from pfm.cache import FeatureCache
from pfm.data import ImageFolder


def _img(p, size=32):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)).save(p)


def test_id_collision_is_refused(tmp_path):
    _img(tmp_path / "a.jpg"); _img(tmp_path / "a.png")
    with pytest.raises(ValueError, match="colliding"):
        ImageFolder(tmp_path, 16)
    _img(tmp_path / "x" / "b.jpg"); _img(tmp_path / "x__b.jpg")
    (tmp_path / "a.png").unlink()
    with pytest.raises(ValueError, match="colliding"):
        ImageFolder(tmp_path, 16)


def test_bad_files_list_is_skipped(tmp_path):
    _img(tmp_path / "ok.jpg")
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "bad_files.txt").write_text("broken.jpg")
    ds = ImageFolder(tmp_path, 16)
    assert len(ds) == 1 and ds[0][0] == "ok"


def test_odd_filenames(tmp_path):
    for n in ["a b.jpg", "ğüş çö.png", "dir with space/İ.JPG", "UPPER.JPEG"]:
        _img(tmp_path / n)
    ds = ImageFolder(tmp_path, 16)
    assert len(ds) == 4
    for i in range(4):
        assert ds[i][1].shape == (3, 16, 16)


def test_cache_size_mismatch_fails_early(tmp_path):
    c = FeatureCache(tmp_path)
    c.write_meta({"size": 224})
    c.check_size(224)
    with pytest.raises(ValueError, match="rebuild the cache"):
        c.check_size(448)
