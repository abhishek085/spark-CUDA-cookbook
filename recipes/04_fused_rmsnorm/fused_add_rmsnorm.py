"""Fuse residual add + RMSNorm into one Triton kernel, and measure what fusion buys.

Eager PyTorch runs ``h = x + r; y = h * rsqrt(mean(h^2) + eps) * w`` as several kernels, each reading and
writing the full activation. The fused kernel reads x, r and w once and writes h and y once. At decode
(few rows) launch overhead dominates; at prefill (thousands of rows) the saved memory traffic does.

This is the pattern for all the small "glue" ops between matmuls (SwiGLU, rotary, gating): write them in
Triton, one program per row, fp32 math inside.

    python recipes/04_fused_rmsnorm/fused_add_rmsnorm.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench as B  # noqa: E402
from common.verify import report_close  # noqa: E402


@triton.jit
def _add_rmsnorm(X, R, W, H, Y, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    tl.store(H + row * D + cols, h.to(H.dtype.element_ty), mask=mask)
    inv = tl.rsqrt(tl.sum(h * h, axis=0) / D + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * D + cols, (h * inv * w).to(Y.dtype.element_ty), mask=mask)


def add_rmsnorm(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """Returns (h, y): the new residual stream and its normalized copy."""

    rows, d = x.shape
    h, y = torch.empty_like(x), torch.empty_like(x)
    block = triton.next_power_of_2(d)
    _add_rmsnorm[(rows,)](x, r, w, h, y, eps, D=d, BLOCK=block, num_warps=min(16, max(1, block // 256)))
    return h, y


def add_rmsnorm_eager(x, r, w, eps: float = 1e-6):
    h = x + r
    hf = h.float()
    return h, (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def main() -> None:
    torch.manual_seed(0)
    compiled = torch.compile(add_rmsnorm_eager)
    for rows, d in [(1, 4096), (8, 4096), (4096, 4096), (4096, 8192)]:
        x = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        w = torch.randn(d, device="cuda", dtype=torch.bfloat16)
        h, y = add_rmsnorm(x, r, w)
        h_ref, y_ref = add_rmsnorm_eager(x, r, w)
        report_close(h, h_ref, atol=0, rtol=0, name=f"h {rows}x{d}")
        report_close(y, y_ref, atol=1e-2, rtol=1e-2, name=f"y {rows}x{d}")
        nbytes = 4 * x.numel() * x.element_size() + w.numel() * w.element_size()   # read x, r; write h, y
        print(B.table([
            B.bench(lambda: add_rmsnorm_eager(x, r, w), name=f"eager {rows}x{d}", bytes=nbytes),
            B.bench(lambda: compiled(x, r, w), name=f"torch.compile {rows}x{d}", bytes=nbytes),
            B.bench(lambda: add_rmsnorm(x, r, w), name=f"triton fused {rows}x{d}", bytes=nbytes),
        ]))
        print()


if __name__ == "__main__":
    main()
