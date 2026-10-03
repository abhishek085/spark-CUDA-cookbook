"""Measure this machine's roofline: achievable memory bandwidth and matmul throughput.

Every later recipe compares itself against these numbers, so they come from this box, not a spec sheet.
Results go to results/roofline.json.

    python recipes/01_roofline/measure.py            # bandwidth sweep + matmul peaks
    python recipes/01_roofline/measure.py --quick    # fewer sizes
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
from common.spark import device_summary  # noqa: E402

HERE = Path(__file__).parent


def bandwidth(sizes_mib: list[int], ext, sms: int) -> list[dict]:
    import torch

    rows = []
    for mib in sizes_mib:
        n = mib * 2**20 // 4
        x = torch.rand(n, device="cuda")
        y = torch.empty_like(x)
        # A sweep over grid sizes: the best is usually a few blocks per SM; the spread itself is worth seeing.
        best: dict[str, B.Result] = {}
        for per_sm in (2, 4, 8, 16):
            blocks = sms * per_sm
            cases = {
                "read": (lambda: ext.read(x, blocks, 256), x.numel() * 4),
                "copy": (lambda: ext.copy(x, y, blocks, 256), 2 * x.numel() * 4),
                "write": (lambda: ext.write(y, 1.0, blocks, 256), y.numel() * 4),
            }
            for kind, (fn, nbytes) in cases.items():
                r = B.bench(fn, name=f"{kind} {mib} MiB x{per_sm}/SM", bytes=nbytes, iters=30)
                if kind not in best or r.gbps > best[kind].gbps:
                    best[kind] = r
        torch_copy = B.bench(lambda: y.copy_(x), name=f"torch copy_ {mib} MiB", bytes=2 * x.numel() * 4, iters=30)
        print(B.table([*best.values(), torch_copy]))
        rows.append({"mib": mib, **{k: round(v.gbps, 1) for k, v in best.items()},
                     "torch_copy": round(torch_copy.gbps, 1)})
        del x, y
        torch.cuda.empty_cache()
    return rows


def matmul_peaks(n: int) -> dict[str, float]:
    import torch

    out = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        r = B.bench(lambda: a @ b, name=f"{name} matmul {n}^3", flops=2 * n**3, cold_l2=False, iters=20)
        print(B.table([r]))
        out[name] = round(r.tflops, 2)
    if hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn"):
        try:
            a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t()   # column-major second operand
            one = torch.ones((), device="cuda")
            r = B.bench(lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
                        name=f"fp8 _scaled_mm {n}^3", flops=2 * n**3, cold_l2=False, iters=20)
            print(B.table([r]))
            out["fp8"] = round(r.tflops, 2)
        except Exception as exc:  # noqa: BLE001 - fp8 support depends on the torch build
            print(f"fp8 matmul skipped: {exc}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--matmul-n", type=int, default=8192)
    args = ap.parse_args()

    info = device_summary()
    ext = load("cookbook_bandwidth", [HERE / "bandwidth.cu"])
    # Small sizes sit in L2; the large ones are what decode sees, since weights never fit in cache.
    sizes = [16, 256, 2048] if args.quick else [4, 16, 64, 256, 1024, 2048, 4096]
    bw = bandwidth(sizes, ext, info["sms"])
    mm = matmul_peaks(args.matmul_n)
    dram = max(r["read"] for r in bw if r["mib"] >= 1024) if any(r["mib"] >= 1024 for r in bw) else bw[-1]["read"]
    result = {"device": info, "dram_read_gbps": dram, "bandwidth": bw, "matmul_tflops": mm}
    out = ROOT / "results" / "roofline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nDRAM read bandwidth (decode's ceiling): {dram} GB/s; matmul TFLOP/s: {mm}")
    print(f"ridge point (bf16): {mm['bf16'] * 1e3 / dram:.0f} FLOP/byte -- kernels below this are memory-bound")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
