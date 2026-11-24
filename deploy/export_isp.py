"""The raw-camera ISP of pfm_runtime._prep_raw as ONE TensorRT engine (Update 167): on the Orin the PyTorch version costs 25 ms per 1640x616
frame (one kernel per step, memory bound); TensorRT fuses the steps and runs them in fp16.
Graph input : mosaic (1,1,H,W) linear 0..1 (black level removed), gains (1,3,1,1) = white balance x digital gain (computed per frame outside).
Graph output: rgb (1,3,H,W) gamma-encoded 0..1 (before undistort / rotation, which stay in PyTorch).
Steps (identical to _prep_raw): odd-row shift (8 px), Malvar-He-Cutler demosaic, gains, colour matrix, chroma denoise, clamp, gamma 1/2.2,
vertical comb filter [1,2,1]/4, guided luma denoise (r 1, eps 5e-4).
Usage: python deploy/export_isp.py [out_dir=deploy/isp] [H=616] [W=1640] [row_shift=8]   (build on the Orin: trtexec --onnx=... --fp16)"""
import sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "deploy/isp"); H = int(sys.argv[2]) if len(sys.argv) > 2 else 616
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1640; SHIFT = int(sys.argv[4]) if len(sys.argv) > 4 else (8 if W == 1640 else 0)


class ISPNet(nn.Module):
    def __init__(self, H, W, shift):
        super().__init__(); self.H, self.W, self.shift = H, W, shift
        m = torch.zeros(4, H, W); m[0, 0::2, 0::2] = 1; m[1, 0::2, 1::2] = 1; m[2, 1::2, 0::2] = 1; m[3, 1::2, 1::2] = 1
        self.register_buffer("m", m[None])
        kG = torch.tensor([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype=torch.float32) / 8
        kRow = torch.tensor([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]], dtype=torch.float32) / 8
        kX = torch.tensor([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], dtype=torch.float32) / 8
        self.register_buffer("k", torch.stack((kG, kRow, kRow.t().contiguous(), kX))[:, None])
        self.register_buffer("ccm", torch.tensor([[1.70, -0.50, -0.20], [-0.35, 1.60, -0.25], [-0.05, -0.55, 1.60]])[:, :, None, None])
        self.register_buffer("kv", torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3, 1).repeat(3, 1, 1, 1))
        rows = torch.arange(H); self.register_buffer("odd", (rows % 2 == 1).float().view(1, 1, H, 1))

    def forward(self, mosaic, gains):
        x = mosaic
        if self.shift:                                   # odd rows 8 px right (even rows unchanged); the first columns repeat the edge
            s = self.shift; shifted = torch.cat((x[..., :1].expand(-1, -1, -1, s), x[..., :-s]), 3)
            x = x * (1 - self.odd) + shifted * self.odd
        c = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.k)
        m = self.m; R_, Gr, Gb, B_ = m[:, 0:1], m[:, 1:2], m[:, 2:3], m[:, 3:4]
        c0, c1, c2, c3 = c[:, 0:1], c[:, 1:2], c[:, 2:3], c[:, 3:4]
        r = x * R_ + c1 * Gr + c2 * Gb + c3 * B_; g = x * (Gr + Gb) + c0 * (R_ + B_); b = x * B_ + c1 * Gb + c2 * Gr + c3 * R_
        rgb = torch.relu(torch.cat((r, g, b), 1)) * gains
        rgb = F.conv2d(rgb, self.ccm)
        yy = rgb.mean(1, keepdim=True); rgb = yy + F.avg_pool2d(rgb - yy, 5, 1, 2, count_include_pad=False)
        rgb = rgb.clamp(0, 1).clamp(min=1e-6).pow(1 / 2.2)
        rgb = F.conv2d(F.pad(rgb, (0, 0, 1, 1), mode="replicate"), self.kv, groups=3)
        Y = rgb.mean(1, keepdim=True); mY = F.avg_pool2d(Y, 3, 1, 1, count_include_pad=False); vY = F.avg_pool2d(Y * Y, 3, 1, 1, count_include_pad=False) - mY * mY
        A = vY / (vY + 5e-4); Bc = mY - A * mY; Yn = F.avg_pool2d(A, 3, 1, 1, count_include_pad=False) * Y + F.avg_pool2d(Bc, 3, 1, 1, count_include_pad=False)
        return (rgb + (Yn - Y)).clamp(0, 1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); net = ISPNet(H, W, SHIFT).eval()
    p = OUT / f"isp_{H}x{W}.onnx"
    torch.onnx.export(net, (torch.rand(1, 1, H, W), torch.ones(1, 3, 1, 1)), str(p), opset_version=17, input_names=["mosaic", "gains"], output_names=["rgb"], dynamo=False)
    print("wrote", p)
