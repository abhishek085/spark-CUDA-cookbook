"""Kernel timing with CUDA events, an optional L2 flush, and achieved GB/s and TFLOP/s.

Rules this follows (each one has burned someone):
- time on the GPU with events, never with time.time() around an async launch;
- warm up first (JIT, autotuning, clocks, lazy allocations);
- report the median and spread, not the best run;
- flush L2 between runs when the working set would otherwise stay cached, or GB/s numbers lie.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Callable, Iterable


@dataclass
class Result:
    name: str
    ms: float            # median
    ms_p10: float
    ms_p90: float
    bytes: float = 0.0   # bytes the kernel must move at minimum
    flops: float = 0.0

    @property
    def gbps(self) -> float:
        return self.bytes / (self.ms * 1e-3) / 1e9 if self.bytes else 0.0

    @property
    def tflops(self) -> float:
        return self.flops / (self.ms * 1e-3) / 1e12 if self.flops else 0.0


_flush_buf = None


def flush_l2() -> None:
    """Overwrite a buffer twice the L2 size so the next kernel starts with a cold cache."""

    global _flush_buf
    import torch

    if _flush_buf is None:
        l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 64 * 2**20
        _flush_buf = torch.empty(2 * l2, dtype=torch.uint8, device="cuda")
    _flush_buf.zero_()


def bench(fn: Callable[[], object], *, name: str = "", warmup: int = 10, iters: int = 100, cold_l2: bool = True,
          bytes: float = 0.0, flops: float = 0.0) -> Result:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        if cold_l2:
            flush_l2()
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    q = statistics.quantiles(times, n=10)
    return Result(name, statistics.median(times), q[0], q[-1], bytes, flops)


def table(results: Iterable[Result], peak_gbps: float = 0.0) -> str:
    rows = [f"{'name':<36} {'median ms':>10} {'p10':>8} {'p90':>8} {'GB/s':>8} {'%peak':>6} {'TFLOP/s':>8}"]
    for r in results:
        pct = f"{100 * r.gbps / peak_gbps:5.1f}%" if peak_gbps and r.gbps else ""
        rows.append(f"{r.name:<36} {r.ms:10.4f} {r.ms_p10:8.4f} {r.ms_p90:8.4f} "
                    f"{r.gbps:8.1f} {pct:>6} {r.tflops:8.2f}")
    return "\n".join(rows)
