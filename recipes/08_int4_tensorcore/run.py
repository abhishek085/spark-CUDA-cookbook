"""Check and benchmark the tensor-core 4-bit matmul against recipe 03, dense bf16 and the fp32 reference.

    python recipes/08_int4_tensorcore/run.py                        # correctness, speed, fused QKV, invariance
    python recipes/08_int4_tensorcore/run.py --shape 4096x4096 --rows 1 16 --sweep-sk
    python recipes/08_int4_tensorcore/run.py --only check           # correctness and invariance only

Sections:
  check      every shape x row count against the fp32 reference (exits non-zero on FAIL)
  speed      tc kernel vs recipe 03's warp GEMV (M <= 8) vs torch bf16, with %peak from results/roofline.json
  fused      Q, K and V as three launches vs one launch on a fused pack
  invariant  row 0 must be bit-identical for every M (what an exact speculative verifier needs)
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "recipes" / "03_int4_gemv"))
sys.path.insert(0, str(HERE))

from common import bench as B  # noqa: E402
from common.build import load  # noqa: E402
from common.verify import bitwise_equal, report_close  # noqa: E402
from quant import matmul_reference, quantize  # noqa: E402
from tc_pack import TCWeight, pack, pack_fused, split_k  # noqa: E402

SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336), (128256, 4096)]
ROWS = [1, 4, 8, 16, 32, 64]


@lru_cache(maxsize=1)
def ext():
    return load("cookbook_qmv_tc", [HERE / "qmv_tc.cu"])


@lru_cache(maxsize=1)
def gemv03():
    return load("cookbook_int4_gemv", [ROOT / "recipes" / "03_int4_gemv" / "gemv.cu"])


def matmul(x: torch.Tensor, w: TCWeight, sk: int | None = None, nt: int = 4) -> torch.Tensor:
    sk = split_k(w.npad, w.k, nt) if sk is None else sk
    return ext().qmv_tc(x, w.words, w.scales, w.biases, w.n, w.gs, sk, nt)


def peak_gbps() -> float:
    f = ROOT / "results" / "roofline.json"
    return json.loads(f.read_text())["dram_read_gbps"] if f.exists() else 0.0


def make(n: int, k: int, gs: int):
    w = torch.randn(n, k, device="cuda") * 0.02
    mlx = quantize(w, gs)
    return mlx, pack(*mlx), w.to(torch.bfloat16)


def check(shapes, rows, gs) -> bool:
    ok = True
    for n, k in shapes:
        mlx, tw, _ = make(n, k, gs)
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = matmul_reference(x, *mlx)
            tol = dict(atol=2e-2 * ref.abs().max().item(), rtol=2e-2)
            ok &= report_close(matmul(x, tw), ref, name=f"tc {n}x{k} M={m} sk={split_k(tw.npad, k)}", **tol)
            ok &= report_close(matmul(x, tw, sk=1), ref, name=f"tc {n}x{k} M={m} sk=1", **tol)
    return ok


def speed(shapes, rows, gs, sweep_sk: bool) -> None:
    peak = peak_gbps()
    g03 = gemv03()
    for n, k in shapes:
        mlx, tw, dense = make(n, k, gs)
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            q_bytes = tw.nbytes() + x.numel() * 2 + m * n * 2
            res = []
            sks = sorted({1, 2, 4, 8, 16, split_k(tw.npad, k)}) if sweep_sk else [split_k(tw.npad, k)]
            for sk in sks:
                if (k // 128) % sk:
                    continue
                for nt in (2, 4) if sweep_sk else (4,):
                    res.append(B.bench(lambda: matmul(x, tw, sk, nt), name=f"tc sk={sk} nt={nt}", bytes=q_bytes))
            if m <= 8:
                res.append(B.bench(lambda: g03.gemv(x, mlx[0], mlx[1], mlx[2], gs, "warp", 256),
                                   name="recipe 03 warp GEMV", bytes=q_bytes))
            res.append(B.bench(lambda: x @ dense.T, name="torch bf16 dense",
                               bytes=dense.numel() * 2 + x.numel() * 2 + m * n * 2))
            print(f"\n{n}x{k}, M={m}")
            print(B.table(res, peak))
        del mlx, tw, dense
        torch.cuda.empty_cache()


def fused(gs: int) -> None:
    """Llama-8B-like attention input: Q 4096, K 1024, V 1024 from a 4096-wide hidden state."""

    k = 4096
    parts = [quantize(torch.randn(n, k, device="cuda") * 0.02, gs) for n in (4096, 1024, 1024)]
    sep = [pack(*p) for p in parts]
    one = pack_fused(parts)
    for m in (1, 16):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        outs = matmul(x, one).split(one.splits, dim=1)
        same = all(report_close(o, matmul_reference(x, *p), atol=2e-2 * o.abs().max().item(), rtol=2e-2,
                                name=f"fused qkv part M={m}") for o, p in zip(outs, parts))
        nbytes = one.nbytes() + x.numel() * 2 + m * one.n * 2
        print(B.table([
            B.bench(lambda: [matmul(x, w) for w in sep], name=f"q, k, v: 3 launches M={m}", bytes=nbytes),
            B.bench(lambda: matmul(x, one), name=f"qkv fused: 1 launch M={m}", bytes=nbytes),
        ], peak_gbps()))
        if not same:
            print("fused output mismatch")


def invariant(gs: int) -> bool:
    ok = True
    for n, k in [(4096, 4096), (4096, 14336)]:
        _, tw, _ = make(n, k, gs)
        x_all = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
        solo = matmul(x_all[:1], tw)
        flags = []
        for m in (1, 2, 3, 8, 15, 16, 17, 32, 64, 128):
            same = bitwise_equal(matmul(x_all[:m], tw)[:1], solo)
            ok &= same
            flags.append(f"M={m}:{'same' if same else 'DIFF'}")
        print(f"tc {n}x{k} row 0: " + "  ".join(flags))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", help="NxK, repeatable")
    ap.add_argument("--rows", type=int, nargs="+", default=ROWS)
    ap.add_argument("--gs", type=int, default=64, choices=[32, 64, 128])
    ap.add_argument("--sweep-sk", action="store_true", help="also try other K splits and 2 tiles per warp")
    ap.add_argument("--only", choices=["check", "speed", "fused", "invariant"])
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shape] if args.shape else SHAPES
    torch.manual_seed(0)

    ok = True
    if args.only in (None, "check"):
        print("== check")
        ok &= check(shapes, args.rows, args.gs)
    if args.only in (None, "invariant"):
        print("\n== invariant")
        ok &= invariant(args.gs)
    if args.only in (None, "speed"):
        print("\n== speed")
        speed(shapes, args.rows, args.gs, args.sweep_sk)
    if args.only in (None, "fused"):
        print("\n== fused")
        fused(args.gs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
