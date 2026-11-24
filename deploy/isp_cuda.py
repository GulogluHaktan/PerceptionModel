"""Raw-camera ISP as hand-written CUDA kernels (Update 167): the PyTorch version of pfm_runtime._prep_raw costs 25 ms per 1640x616 frame on
the Orin (one memory-bound kernel per op) and a TensorRT engine of it 13.6 ms. Five small kernels, compiled once with torch.utils.cpp_extension:
  1 demosaic : black level, row-class shift (8*(y%4) px), Malvar-He-Cutler 5x5, gains (WB x digital), colour matrix -> linear RGB
  2 chroma   : 5x5 chroma denoise (box mean of colour minus luma), clamp, gamma 1/2.2
  3 comb     : vertical [1,2,1]/4
  4 guided-a : 3x3 mean / variance of luma -> A, B (eps 5e-4)
  5 guided-b : 3x3 means of A, B -> luma correction, clamp -> output
Same maths as _prep_raw (borders: reflect in the demosaic, valid-only means elsewhere). Gains come from cheap plane statistics of the raw frame.
    isp = CudaISP(); rgb = isp(raw_uint16_numpy, out=None)   # (1,3,H,W) float32 on cuda; isp.lin = linear grey level (for auto exposure)"""
import os
import numpy as np
import torch

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define BL 4096.f
#define SC (1.f / (65535.f - 4096.f))
__device__ __forceinline__ int refl(int i, int n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; return i; }
__device__ __forceinline__ float rawpx(const short* raw, int H, int W, int y, int x, int sh) {
    y = refl(y, H); x = refl(x, W);
    int xs = x - sh * (y & 3); if (xs < 0) xs = 0;      // x2-binned full-width readout: row y comes out 8*(y%4) px left
    float v = (float)(unsigned short)raw[y * W + xs];
    return fmaxf(v - BL, 0.f) * SC;
}
__constant__ float KG[25] = {0,0,-1,0,0, 0,0,2,0,0, -1,2,4,2,-1, 0,0,2,0,0, 0,0,-1,0,0};
__constant__ float KR[25] = {0,0,.5f,0,0, 0,-1,0,-1,0, -1,4,5,4,-1, 0,-1,0,-1,0, 0,0,.5f,0,0};
__constant__ float KC[25] = {0,0,-1,0,0, 0,-1,4,-1,0, .5f,0,5,0,.5f, 0,-1,4,-1,0, 0,0,-1,0,0};
__constant__ float KX[25] = {0,0,-1.5f,0,0, 0,2,0,2,0, -1.5f,0,6,0,-1.5f, 0,2,0,2,0, 0,0,-1.5f,0,0};
__global__ void k_demosaic(const short* raw, float* out, int H, int W, int sh, float g0, float g1, float g2,
                           float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
    float cg = 0, cr = 0, cc = 0, cx = 0, p = 0;
    #pragma unroll
    for (int dy = -2; dy <= 2; dy++)
    #pragma unroll
    for (int dx = -2; dx <= 2; dx++) {
        int k = (dy + 2) * 5 + dx + 2; float v = rawpx(raw, H, W, y + dy, x + dx, sh);
        cg += KG[k] * v; cr += KR[k] * v; cc += KC[k] * v; cx += KX[k] * v; if (k == 12) p = v;
    }
    cg *= .125f; cr *= .125f; cc *= .125f; cx *= .125f;
    bool ey = !(y & 1), ex = !(x & 1); float r, g, b;
    if (ey && ex)       { r = p;  g = cg; b = cx; }      // R site
    else if (ey && !ex) { r = cr; g = p;  b = cc; }      // Gr (red row)
    else if (!ey && ex) { r = cc; g = p;  b = cr; }      // Gb (blue row)
    else                { r = cx; g = cg; b = p;  }      // B site
    r = fmaxf(r, 0.f) * g0; g = fmaxf(g, 0.f) * g1; b = fmaxf(b, 0.f) * g2;
    int n = H * W, i = y * W + x;
    out[i] = m00 * r + m01 * g + m02 * b; out[n + i] = m10 * r + m11 * g + m12 * b; out[2 * n + i] = m20 * r + m21 * g + m22 * b;
}
__global__ void k_chroma(const float* in, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
    int n = H * W; float s0 = 0, s1 = 0, s2 = 0; int c = 0;
    for (int dy = -2; dy <= 2; dy++) { int yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (int dx = -2; dx <= 2; dx++) { int xx = x + dx; if (xx < 0 || xx >= W) continue;
            int j = yy * W + xx; float a = in[j], b = in[n + j], d = in[2 * n + j], m = (a + b + d) * (1.f / 3.f);
            s0 += a - m; s1 += b - m; s2 += d - m; c++; } }
    int i = y * W + x; float Y = (in[i] + in[n + i] + in[2 * n + i]) * (1.f / 3.f), ic = 1.f / c;
    out[i] = __powf(fminf(fmaxf(Y + s0 * ic, 1e-6f), 1.f), 1.f / 2.2f);
    out[n + i] = __powf(fminf(fmaxf(Y + s1 * ic, 1e-6f), 1.f), 1.f / 2.2f);
    out[2 * n + i] = __powf(fminf(fmaxf(Y + s2 * ic, 1e-6f), 1.f), 1.f / 2.2f);
}
__global__ void k_comb(const float* in, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
    int n = H * W, i = y * W + x;
    for (int ch = 0; ch < 3; ch++) { const float* p = in + ch * n;
        out[ch * n + i] = (y == 0 || y == H - 1) ? p[i] : .25f * p[i - W] + .5f * p[i] + .25f * p[i + W]; }
}
__global__ void k_guided_a(const float* in, float* ab, int H, int W, float eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
    int n = H * W; float s = 0, ss = 0; int c = 0;
    for (int dy = -1; dy <= 1; dy++) { int yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (int dx = -1; dx <= 1; dx++) { int xx = x + dx; if (xx < 0 || xx >= W) continue;
            int j = yy * W + xx; float Y = (in[j] + in[n + j] + in[2 * n + j]) * (1.f / 3.f); s += Y; ss += Y * Y; c++; } }
    float m = s / c, v = ss / c - m * m, A = v / (v + eps); int i = y * W + x; ab[i] = A; ab[n + i] = m - A * m;
}
__global__ void k_guided_b(const float* in, const float* ab, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
    int n = H * W; float sa = 0, sb = 0; int c = 0;
    for (int dy = -1; dy <= 1; dy++) { int yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (int dx = -1; dx <= 1; dx++) { int xx = x + dx; if (xx < 0 || xx >= W) continue;
            int j = yy * W + xx; sa += ab[j]; sb += ab[n + j]; c++; } }
    int i = y * W + x; float Y = (in[i] + in[n + i] + in[2 * n + i]) * (1.f / 3.f), d = (sa / c) * Y + sb / c - Y;
    for (int ch = 0; ch < 3; ch++) out[ch * n + i] = fminf(fmaxf(in[ch * n + i] + d, 0.f), 1.f);
}
void run(torch::Tensor raw, torch::Tensor t1, torch::Tensor t2, torch::Tensor ab, torch::Tensor out, int sh,
         double g0, double g1, double g2, std::vector<double> ccm, double eps) {
    int H = raw.size(0), W = raw.size(1); dim3 b(32, 8), g((W + 31) / 32, (H + 7) / 8);
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    k_demosaic<<<g, b, 0, s>>>((const short*)raw.data_ptr(), t1.data_ptr<float>(), H, W, sh, g0, g1, g2,
        ccm[0], ccm[1], ccm[2], ccm[3], ccm[4], ccm[5], ccm[6], ccm[7], ccm[8]);
    k_chroma<<<g, b, 0, s>>>(t1.data_ptr<float>(), t2.data_ptr<float>(), H, W);
    k_comb<<<g, b, 0, s>>>(t2.data_ptr<float>(), t1.data_ptr<float>(), H, W);
    k_guided_a<<<g, b, 0, s>>>(t1.data_ptr<float>(), ab.data_ptr<float>(), H, W, (float)eps);
    k_guided_b<<<g, b, 0, s>>>(t1.data_ptr<float>(), ab.data_ptr<float>(), out.data_ptr<float>(), H, W);
}
"""
_CPP = "void run(torch::Tensor raw, torch::Tensor t1, torch::Tensor t2, torch::Tensor ab, torch::Tensor out, int sh, double g0, double g1, double g2, std::vector<double> ccm, double eps);"
_mod = None


def _load():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline("pfm_isp_cuda", cpp_sources=_CPP, cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + _SRC, functions=["run"],
                           extra_cuda_cflags=["-O3", "--use_fast_math"], build_directory=os.environ.get("PFM_ISP_BUILD") or None, verbose=False)
    return _mod


CCM = [1.70, -0.50, -0.20, -0.35, 1.60, -0.25, -0.05, -0.55, 1.60]


class CudaISP:
    def __init__(self, dev="cuda", row_shift=None):
        self.dev, self.row_shift = dev, row_shift; self.bufs = None; self.wb = np.ones(3); self.dgain = 1.0; self.lin = None; _load()

    def __call__(self, frame, out=None):
        H, W = frame.shape
        if self.bufs is None or self.bufs[0].shape[-2:] != (H, W):
            self.bufs = (torch.empty(3, H, W, device=self.dev), torch.empty(3, H, W, device=self.dev), torch.empty(2, H, W, device=self.dev))
            # no host stalls (Update 167): pinned staging (2 slots, each guarded by an event) -> async upload; plane stats copied back async and
            # used one frame later (the gains are an EMA anyway). Before: pageable copy + .cpu() sync = ~3.5 ms of host wait per frame.
            self.pin = [torch.empty(H, W, dtype=torch.int16).pin_memory() for _ in range(2)]; self.pev = [None, None]; self.pk = 0
            self.raw = torch.empty(H, W, dtype=torch.int16, device=self.dev)
            self.spin = torch.empty(3, dtype=torch.float32).pin_memory(); self.sev = None
        k = self.pk; self.pk ^= 1
        if self.pev[k] is not None: self.pev[k].synchronize()            # the upload that used this staging slot two frames ago
        self.pin[k].numpy()[...] = np.asarray(frame).view(np.int16)
        raw = self.raw; raw.copy_(self.pin[k], non_blocking=True); ev = torch.cuda.Event(); ev.record(); self.pev[k] = ev
        # gains from plane statistics of the raw frame (1/16 subsample): grey world (green = 1) and digital gain to an 18 % grey, both EMA
        prev = None
        if self.sev is not None: self.sev.synchronize(); prev = self.spin.numpy().astype(np.float64).copy()   # last frame's stats (long done)
        pl = lambda r0, c0: raw[r0::2, c0::2][::4, ::4].to(torch.int32).bitwise_and_(0xFFFF).float().sub_(4096).clamp_(min=0).mean() / (65535 - 4096)
        r, g1, g2, b = pl(0, 0), pl(0, 1), pl(1, 0), pl(1, 1)          # Bayer planes FIRST, then subsample (a plain ::4 hits only R sites)
        self.spin.copy_(torch.stack((r, (g1 + g2) / 2, b)), non_blocking=True); self.sev = torch.cuda.Event(); self.sev.record()
        m = prev if prev is not None else (self.sev.synchronize(), self.spin.numpy().astype(np.float64).copy())[1]
        self.wb = 0.9 * self.wb + 0.1 * np.clip(m[1] / np.maximum(m, 1e-4), 0.4, 3.0); self.lin = float((m * self.wb).mean())
        self.dgain = 0.9 * self.dgain + 0.1 * float(np.clip(0.18 / max(self.lin, 1e-4), 0.5, 16.0))
        gn = self.wb * self.dgain; sh = self.row_shift if self.row_shift is not None else (8 if W == 1640 else 0)
        if out is None: out = torch.empty(1, 3, H, W, device=self.dev)
        _mod.run(raw, self.bufs[0], self.bufs[1], self.bufs[2], out, int(sh), float(gn[0]), float(gn[1]), float(gn[2]), CCM, 5e-4)
        return out
