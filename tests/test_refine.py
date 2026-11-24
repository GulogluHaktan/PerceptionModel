import math
import torch
from pfm.refine import CornerRefiner, P, sample_patches, to_image, to_patch, refine_gates


def test_patch_roundtrip_and_canonical_direction():
    torch.manual_seed(0)
    n = 16
    corner = torch.rand(n, 2) * 200 + 50
    centroid = corner + torch.randn(n, 2) * 40
    size = torch.rand(n) * 300 + 10
    img = torch.zeros(1, 3, 320, 320)
    _, s, R = sample_patches(img, torch.zeros(n, dtype=torch.long), corner, centroid, size)
    tgt = corner + torch.randn(n, 2) * 5
    assert torch.allclose(to_image(to_patch(tgt, corner, s, R), corner, s, R), tgt, atol=1e-4)
    d = to_patch(centroid, corner, s, R)            # centroid must land on the bottom-right diagonal of the patch
    assert torch.allclose(d[:, 0], d[:, 1], atol=1e-3) and (d[:, 0] > 0).all()


def test_sampling_hits_the_marked_pixel():
    img = torch.zeros(1, 3, 200, 200)
    img[0, :, 120, 80] = 1.0                        # pixel (x=80, y=120)
    corner = torch.tensor([[75.0, 110.0]]); centroid = torch.tensor([[150.0, 30.0]]); size = torch.tensor([50.0])
    patches, s, R = sample_patches(img, torch.zeros(1, dtype=torch.long), corner, centroid, size)
    k = patches[0, 0].flatten().argmax(); py, px = divmod(int(k), P)
    q = torch.tensor([[px - (P - 1) / 2, py - (P - 1) / 2]])
    back = to_image(q, corner, s, R)
    assert (back - torch.tensor([[80.0, 120.0]])).norm() < 1.0


def test_model_shapes_and_refine_gates():
    m = CornerRefiner().eval()
    off, vis, h = m(torch.rand(3, 3, P, P))
    assert off.shape == (3, 2) and vis.shape == (3,) and h.shape == (3, P * P)
    img = torch.rand(2, 3, 120, 160)
    gates = [torch.tensor([[10.0, 10, 50, 10, 50, 40, 10, 40]]), torch.zeros(0, 8)]
    out = refine_gates(m, img, gates)
    assert out[0].shape == (1, 8) and len(out[1]) == 0
