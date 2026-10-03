"""Repack MLX 4-bit weights, once at load time, into the order the tensor-core kernel consumes them.

The kernel uses ``mma.sync.m16n8k16`` (bf16 in, fp32 accumulate). For one 8-column weight tile and one
16-deep k step, lane L (g = L // 4, t = L % 4) must hand the tensor core the weights of column g at
k = 2t, 2t+1 (B register 0) and k = 2t+8, 2t+9 (B register 1). This packing puts exactly those codes
next to each other so the kernel does one 16-byte load per lane per 128-k chunk and no shuffling:

    words[T][C][lane][v]   int32, T = 8-column tile, C = 128-k chunk, v = 0..3
    word v covers k16 steps 2v and 2v+1. Nibble p holds pair (p & 3), element (p >> 2):
        pair 0: step 2v,   register 0      pair 2: step 2v+1, register 0
        pair 1: step 2v,   register 1      pair 3: step 2v+1, register 1
    so ``(word >> 4p) & 0x000F000F`` is a register's two codes in bf16 lane positions.

Scales and biases are stored transposed, (K / gs, Npad), because the kernel's output fragment holds
columns 2t and 2t+1, which then read one aligned bf16 pair.

N is padded to a multiple of 32 (four tiles per warp) with zero codes, scales and biases.
Pure PyTorch, runs on CPU, so tests can check the layout without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_int4_gemv"))

from quant import unpack  # noqa: E402

N_ALIGN = 32
K_CHUNK = 128


@dataclass
class TCWeight:
    words: torch.Tensor     # (Npad / 8, K / 128, 32, 4) int32
    scales: torch.Tensor    # (K / gs, Npad) bf16
    biases: torch.Tensor    # (K / gs, Npad) bf16
    n: int
    k: int
    gs: int
    splits: tuple[int, ...] = ()    # output widths when several projections were packed together

    @property
    def npad(self) -> int:
        return self.words.shape[0] * 8

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.words, self.scales, self.biases))


def _lane_offsets() -> tuple[torch.Tensor, torch.Tensor]:
    """(32,) column-in-tile g, and (32, 4, 8) k-in-chunk for every lane, word and nibble."""

    lane = torch.arange(32)
    g, t = lane >> 2, lane & 3
    v = torch.arange(4)[None, :, None]
    p = torch.arange(8)[None, None, :]
    pi, e = p & 3, p >> 2
    step = 2 * v + (pi >> 1)
    reg = pi & 1
    kk = 16 * step + 8 * reg + 2 * t[:, None, None] + e
    return g, kk


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """(Npad, K) codes in [0, 15] -> (Npad / 8, K / 128, 32, 4) int32 words."""

    npad, k = codes.shape
    if npad % 8 or k % K_CHUNK:
        raise ValueError(f"need N % 8 == 0 and K % {K_CHUNK} == 0, got {npad}x{k}")
    g, kk = _lane_offsets()
    g, kk = g.to(codes.device), kk.to(codes.device)
    q = codes.to(torch.int64).reshape(npad // 8, 8, k // K_CHUNK, K_CHUNK).permute(0, 2, 1, 3)   # (T, C, 8, 128)
    picked = q[:, :, g[:, None, None], kk]                                                     # (T, C, 32, 4, 8)
    shifts = (torch.arange(8, device=codes.device, dtype=torch.int64) * 4)
    words = (picked << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()


def unpack_codes(words: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_codes``, decoded the way the kernel decodes (see ``pair`` in qmv_tc.cu)."""

    tiles, chunks = words.shape[:2]
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.empty(tiles, 8, chunks, K_CHUNK, dtype=torch.int64, device=words.device)
    lane = torch.arange(32, device=words.device)
    g, t = lane >> 2, lane & 3
    for step in range(8):
        word = w[..., step >> 1]                                      # (T, C, 32)
        for reg in range(2):
            p = (step & 1) * 2 + reg
            for e, shift in ((0, 4 * p), (1, 4 * p + 16)):
                kk = 16 * step + 8 * reg + 2 * t + e
                out[:, g, :, kk] = ((word >> shift) & 0xF).permute(2, 0, 1)
    return out.reshape(tiles * 8, chunks * K_CHUNK)


def pack(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, chunk_rows: int = 8192) -> TCWeight:
    """MLX layout (N, K/8) int32 + (N, K/gs) scales/biases -> ``TCWeight``."""

    n, k = packed.shape[0], packed.shape[1] * 8
    gs = k // scales.shape[1]
    if gs not in (32, 64, 128):
        raise ValueError(f"group size {gs} unsupported (32, 64 or 128)")
    if k % K_CHUNK:
        raise ValueError(f"K={k} must be a multiple of {K_CHUNK}")
    npad = -(-n // N_ALIGN) * N_ALIGN
    words = torch.empty(npad // 8, k // K_CHUNK, 32, 4, dtype=torch.int32, device=packed.device)
    for start in range(0, npad, chunk_rows):
        stop = min(start + chunk_rows, npad)
        codes = torch.zeros(stop - start, k, dtype=torch.int64, device=packed.device)
        if start < n:
            codes[:min(stop, n) - start] = unpack(packed[start:min(stop, n)])
        words[start // 8:stop // 8] = pack_codes(codes)

    def pad_t(t: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(npad, t.shape[1], dtype=torch.bfloat16, device=t.device)
        full[:n] = t
        return full.T.contiguous()

    return TCWeight(words, pad_t(scales), pad_t(biases), n, k, gs)


def pack_fused(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> TCWeight:
    """Several projections of the same input (Q, K, V; or gate, up) packed as one weight: one launch serves all.

    Split the output with ``out.split(w.splits, dim=1)``.
    """

    packed = torch.cat([p[0] for p in parts])
    scales = torch.cat([p[1] for p in parts])
    biases = torch.cat([p[2] for p in parts])
    w = pack(packed, scales, biases)
    w.splits = tuple(p[0].shape[0] for p in parts)
    return w


def split_k(npad: int, k: int, nt: int = 4, target: int = 512, max_sk: int = 16) -> int:
    """K slices for a weight: a function of its shape only, never of the row count, so rows stay invariant.

    Small layers have too few column tiles to fill the GPU; slicing K multiplies the warps.
    """

    items, chunks, sk = npad // (8 * nt), k // K_CHUNK, 1
    while sk * 2 <= max_sk and items * sk < target and chunks % (sk * 2) == 0 and chunks // (sk * 2) >= 2:
        sk *= 2
    return sk
