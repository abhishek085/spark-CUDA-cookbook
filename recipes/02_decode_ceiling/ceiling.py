"""Upper bounds for LLM decode and prefill speed from bytes and FLOPs, before writing any kernel.

Decode reads every active weight (plus the KV cache) once per forward pass, so for one stream:

    tok/s <= bandwidth / (weight_bytes + kv_bytes_per_token * context)

Speculative decoding verifies several drafted tokens in one forward pass. If a round commits
``accepted`` tokens on average and the draft costs ``draft_cost`` of a target pass:

    tok/s <= accepted * bandwidth / (bytes_per_pass * (1 + draft_cost))

Concurrent streams share one weight read per pass, which is why batching raises aggregate throughput.

    python recipes/02_decode_ceiling/ceiling.py --params 27 --bits 4 --group 64 --bw 273
    python recipes/02_decode_ceiling/ceiling.py --params 30 --active 3 --bits 4 --bw 273 --accepted 3
    python recipes/02_decode_ceiling/ceiling.py --params 8 --bw-from-results     # uses recipe 01's measurement
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path


def bits_per_weight(bits: int, group: int, scale_bits: int = 16, has_bias: bool = True) -> float:
    """Storage per weight with group-wise scales (and affine biases): 4-bit, g64, bf16 scale+bias = 4.5 bits."""

    if bits >= 16 or group <= 0:
        return float(bits)
    return bits + scale_bits * (2 if has_bias else 1) / group


@dataclass
class Model:
    params_b: float            # total parameters, billions
    active_b: float            # parameters read per token (== params_b for dense; MoE active params)
    bits: int = 4
    group: int = 64
    layers: int = 0            # KV cache shape; 0 ignores the KV term
    kv_heads: int = 0
    head_dim: int = 0
    kv_bytes: float = 2.0      # 2 = bf16, 1 = fp8
    attn_layer_fraction: float = 1.0   # hybrid models (linear attention / SSM layers) keep KV in a fraction only

    def weight_bytes_per_pass(self) -> float:
        return self.active_b * 1e9 * bits_per_weight(self.bits, self.group) / 8

    def kv_bytes_per_token(self) -> float:
        return 2 * self.layers * self.attn_layer_fraction * self.kv_heads * self.head_dim * self.kv_bytes


def decode_ceiling(m: Model, bw_gbps: float, context: int = 0, streams: int = 1, accepted: float = 1.0,
                   draft_cost: float = 0.0) -> dict[str, float]:
    """Best-case tokens/s, assuming the pass is purely memory-bound and runs at ``bw_gbps``."""

    weights = m.weight_bytes_per_pass()
    kv = m.kv_bytes_per_token() * context * streams        # each stream reads its own cache
    seconds = (weights + kv) * (1 + draft_cost) / (bw_gbps * 1e9)
    per_stream = accepted / seconds
    return {"weight_gb": weights / 1e9, "kv_gb": kv / 1e9, "ms_per_pass": seconds * 1e3,
            "tok_s_per_stream": per_stream, "tok_s_total": per_stream * streams}


def prefill_ceiling(m: Model, tflops: float, efficiency: float = 0.6) -> float:
    """Prompt tokens/s when compute-bound: 2 FLOPs per active parameter per token, at ``efficiency`` of peak."""

    return tflops * 1e12 * efficiency / (2 * m.active_b * 1e9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", type=float, required=True, help="total params, billions")
    ap.add_argument("--active", type=float, help="active params per token (MoE), billions")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--bw", type=float, help="memory bandwidth GB/s (measured, not spec)")
    ap.add_argument("--bw-from-results", action="store_true", help="read recipe 01's results/roofline.json")
    ap.add_argument("--tflops", type=float, help="dense matmul TFLOP/s for the prefill estimate")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--kv-heads", type=int, default=0)
    ap.add_argument("--head-dim", type=int, default=0)
    ap.add_argument("--kv-bytes", type=float, default=2.0)
    ap.add_argument("--attn-fraction", type=float, default=1.0)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--accepted", type=float, default=1.0, help="mean tokens committed per verify pass")
    ap.add_argument("--draft-cost", type=float, default=0.0, help="draft cost as a fraction of a target pass")
    a = ap.parse_args()

    bw, tflops = a.bw, a.tflops
    if a.bw_from_results:
        r = json.loads((Path(__file__).resolve().parents[2] / "results" / "roofline.json").read_text())
        bw = bw or r["dram_read_gbps"]
        tflops = tflops or r["matmul_tflops"].get("bf16")
    if not bw:
        ap.error("give --bw or --bw-from-results")
    m = Model(a.params, a.active or a.params, a.bits, a.group, a.layers, a.kv_heads, a.head_dim, a.kv_bytes,
              a.attn_fraction)
    plain = decode_ceiling(m, bw, a.context, a.streams)
    spec = decode_ceiling(m, bw, a.context, a.streams, a.accepted, a.draft_cost)
    print(f"{bits_per_weight(a.bits, a.group):.2f} bits/weight -> {plain['weight_gb']:.2f} GB read per pass, "
          f"KV {plain['kv_gb']:.2f} GB at {a.context} tokens x {a.streams} streams, {bw:.0f} GB/s")
    print(f"serial decode ceiling:      {plain['tok_s_per_stream']:7.1f} tok/s per stream, "
          f"{plain['tok_s_total']:7.1f} total")
    if a.accepted != 1.0 or a.draft_cost:
        print(f"speculative ceiling:        {spec['tok_s_per_stream']:7.1f} tok/s per stream, "
              f"{spec['tok_s_total']:7.1f} total  ({a.accepted} accepted/pass, draft cost {a.draft_cost:.0%})")
    if tflops:
        print(f"prefill ceiling (60% of {tflops} TFLOP/s): {prefill_ceiling(m, tflops):,.0f} tok/s")


if __name__ == "__main__":
    main()
