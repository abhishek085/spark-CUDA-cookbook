"""Correctness checks: closeness to a reference, and bitwise equality for invariance tests."""

from __future__ import annotations


def report_close(out, ref, *, atol: float, rtol: float, name: str = "") -> bool:
    """Print max abs/rel error against ``ref`` (compared in fp32) and return whether it is within tolerance."""

    import torch

    o, r = out.float(), ref.float()
    diff = (o - r).abs()
    ok = bool(torch.all(diff <= atol + rtol * r.abs()))
    rel = (diff / r.abs().clamp_min(1e-6)).max().item()
    print(f"{name:<36} max_abs={diff.max().item():.3e} max_rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def bitwise_equal(a, b) -> bool:
    """True when two tensors hold exactly the same bits (NaNs with the same payload count as equal)."""

    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    ints = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return bool(torch.equal(a.contiguous().view(ints), b.contiguous().view(ints)))
