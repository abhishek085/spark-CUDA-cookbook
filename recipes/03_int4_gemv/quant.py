"""Affine 4-bit group quantization in MLX's layout, and a plain PyTorch reference matmul.

A weight ``W`` of shape (N, K) is split along K into groups of ``gs`` (64 by default). Each group stores
4-bit codes ``q`` in [0, 15] plus a bf16 ``scale`` and ``bias``, and dequantizes as ``w = q * scale + bias``.
Codes are packed eight to an int32, lowest nibble first: nibble j of word i is column ``8 * i + j``.
This is the layout MLX 4-bit checkpoints store, so the kernels can read those weights directly.

Runs on CPU too, so the tests do not need a GPU.
"""

from __future__ import annotations

import torch


def quantize(w: torch.Tensor, gs: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(N, K) float -> packed int32 (N, K/8), bf16 scales (N, K/gs), bf16 biases (N, K/gs)."""

    n, k = w.shape
    if k % gs or gs % 8:
        raise ValueError(f"K={k} must be a multiple of the group size {gs}, itself a multiple of 8")
    g = w.float().reshape(n, k // gs, gs)
    lo, hi = g.amin(-1), g.amax(-1)
    # Round scale and bias to bf16 first and quantize against the rounded values, so dequantize() is
    # exactly what was quantized against.
    scales = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    biases = lo.to(torch.bfloat16)
    q = torch.round((g - biases.float()[..., None]) / scales.float()[..., None]).clamp(0, 15).to(torch.int64)
    return pack(q.reshape(n, k)), scales, biases


def pack(q: torch.Tensor) -> torch.Tensor:
    """(N, K) codes in [0, 15] -> (N, K/8) int32, lowest nibble first."""

    n, k = q.shape
    shifts = torch.arange(8, device=q.device, dtype=torch.int64) * 4
    words = (q.to(torch.int64).reshape(n, k // 8, 8) << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)   # same bits as uint32


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """(N, K/8) int32 -> (N, K) int64 codes."""

    n, k8 = packed.shape
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * 4
    return ((packed.to(torch.int64)[..., None] >> shifts) & 0xF).reshape(n, k8 * 8)


def dequantize(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Packed weights -> (N, K) fp32."""

    q = unpack(packed).float()
    n, k = q.shape
    gs = k // scales.shape[1]
    q = q.reshape(n, k // gs, gs)
    return (q * scales.float()[..., None] + biases.float()[..., None]).reshape(n, k)


def matmul_reference(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                     biases: torch.Tensor) -> torch.Tensor:
    """``x @ dequantize(W).T`` in fp32, the ground truth for the kernels."""

    return x.float() @ dequantize(packed, scales, biases).T
