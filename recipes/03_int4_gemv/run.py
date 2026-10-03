"""Check and benchmark the 4-bit GEMV against the fp32 reference and a dense bf16 matmul.

    python recipes/03_int4_gemv/run.py                     # typical 8B-class projection shapes, M = 1, 4, 8
    python recipes/03_int4_gemv/run.py --shape 5120x5120 --rows 1 --threads 128 256 512

"%peak" is against results/roofline.json from recipe 01 (run that first). The goal for a decode GEMV is
to approach that read bandwidth; 4-bit should beat dense bf16 by close to the byte ratio (16 / 4.5 = 3.6x).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import bench as B  # noqa: E402
from common.build import load  # noqa: E402
from common.verify import report_close  # noqa: E402
from quant import matmul_reference, quantize  # noqa: E402

HERE = Path(__file__).parent
# (N, K) of an 8B-class layer: q/o proj, gate/up, down, and a large vocab head
SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336), (128256, 4096)]


def peak_gbps() -> float:
    f = ROOT / "results" / "roofline.json"
    return json.loads(f.read_text())["dram_read_gbps"] if f.exists() else 0.0


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", help="NxK, repeatable (default: 8B-class shapes)")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--threads", type=int, nargs="+", default=[256])
    ap.add_argument("--skip-naive", action="store_true", help="the naive kernel is slow on the vocab head")
    args = ap.parse_args()

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shape] if args.shape else SHAPES
    ext = load("cookbook_int4_gemv", [HERE / "gemv.cu"])
    peak = peak_gbps()
    torch.manual_seed(0)
    all_ok = True
    for n, k in shapes:
        w = torch.randn(n, k, device="cuda") * 0.02
        packed, scales, biases = quantize(w, args.gs)
        dense = w.to(torch.bfloat16)
        del w
        for m in args.rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = matmul_reference(x, packed, scales, biases)
            tol = dict(atol=2e-2 * ref.abs().max().item(), rtol=2e-2)
            q_bytes = packed.numel() * 4 + scales.numel() * 2 * 2 + x.numel() * 2 + m * n * 2
            results = []
            variants = [("warp", t) for t in args.threads]
            if not args.skip_naive:
                variants.insert(0, ("naive", 256))
            for variant, threads in variants:
                out = ext.gemv(x, packed, scales, biases, args.gs, variant, threads)
                all_ok &= report_close(out, ref, name=f"{variant}/{threads} {n}x{k} M={m}", **tol)
                results.append(B.bench(lambda: ext.gemv(x, packed, scales, biases, args.gs, variant, threads),
                                       name=f"int4 {variant}/{threads} M={m}", bytes=q_bytes))
            results.append(B.bench(lambda: x @ dense.T, name=f"bf16 dense torch M={m}",
                                   bytes=dense.numel() * 2 + x.numel() * 2 + m * n * 2))
            print(f"\n{n}x{k}, M={m}")
            print(B.table(results, peak))
        del packed, scales, biases, dense
        torch.cuda.empty_cache()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
