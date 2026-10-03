"""Does a row's result depend on how many other rows run with it?

Speculative decoding verifies k drafted tokens in one batched pass and keeps a draft only when it equals
what serial decoding would produce. If the batched kernels give row 0 different bits than a 1-row call
(cuBLAS picks different algorithms and split-K by shape), drafts get rejected or, worse, the output drifts
from serial decoding. TensorFold makes every verify-path kernel row-invariant for this reason.

This checks, for each row count M, whether row 0 is bit-identical to the M = 1 result, for:
  - torch.matmul (cuBLAS), bf16
  - the cookbook's 4-bit warp GEMV (recipe 03), which computes each row independently by construction

    python recipes/06_batch_invariance/check_invariance.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "recipes" / "03_int4_gemv"))

from common.build import load  # noqa: E402
from common.verify import bitwise_equal  # noqa: E402
from quant import quantize  # noqa: E402


def sweep(name: str, fn, x_all: torch.Tensor, rows: list[int]) -> None:
    solo = fn(x_all[:1])
    flags = []
    for m in rows:
        out = fn(x_all[:m])
        same = bitwise_equal(out[:1], solo)
        diff = (out[:1].float() - solo.float()).abs().max().item()
        flags.append(f"M={m}:{'same' if same else f'DIFF({diff:.1e})'}")
    print(f"{name:<28} " + "  ".join(flags))


def main() -> None:
    torch.manual_seed(0)
    gemv = load("cookbook_int4_gemv", [ROOT / "recipes" / "03_int4_gemv" / "gemv.cu"])
    for n, k in [(4096, 4096), (14336, 4096), (4096, 14336)]:
        w = torch.randn(n, k, device="cuda") * 0.02
        dense = w.to(torch.bfloat16)
        packed, scales, biases = quantize(w)
        x_all = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
        print(f"\n{n}x{k}: is row 0 identical to its M=1 result?")
        sweep("torch.matmul bf16", lambda x: x @ dense.T, x_all, [1, 2, 4, 8, 16, 32, 64, 128])
        sweep("int4 warp GEMV", lambda x: gemv.gemv(x.contiguous(), packed, scales, biases, 64, "warp", 256),
              x_all, [1, 2, 4, 8])
    print("\nAny DIFF on the verify path means drafts are checked against different numbers than serial "
          "decoding produces. Fix it with a shape-only split-K, a fixed reduction order, and one kernel "
          "for both serial and batched rows.")


if __name__ == "__main__":
    main()
