"""Capture a decode step in a CUDA graph and measure the launch overhead it removes.

A small model's decode step is hundreds of tiny kernels. On the CPU side each launch costs a few
microseconds, and when the GPU work per kernel is shorter than that the GPU sits idle between them.
A CUDA graph records the whole sequence once and replays it with one launch.

Rules for capture: fixed shapes, static input/output buffers (copy new data in, read results out), no
host syncs or .item() inside, no allocation that changes between replays. Capture one graph per row
count you serve (1, 2, 4, 8 ... draft widths), the way serving engines do.

    python recipes/05_cuda_graphs/decode_graph.py --layers 32 --dim 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import bench as B  # noqa: E402
from common.verify import bitwise_equal  # noqa: E402


class ToyLayer(torch.nn.Module):
    """RMSNorm, a fused QKV-sized projection, an output projection and a SwiGLU MLP: decode-shaped work."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm1, self.norm2 = torch.nn.RMSNorm(dim), torch.nn.RMSNorm(dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.o = torch.nn.Linear(dim, dim, bias=False)
        self.gate_up = torch.nn.Linear(dim, 2 * hidden, bias=False)
        self.down = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        h = h + self.o(q * torch.sigmoid(k) + v)          # stand-in for attention: same launch pattern
        g, u = self.gate_up(self.norm2(h)).chunk(2, dim=-1)
        return h + self.down(torch.nn.functional.silu(g) * u)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=5632)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()

    model = torch.nn.Sequential(*[ToyLayer(args.dim, args.hidden) for _ in range(args.layers)])
    model = model.to("cuda", torch.bfloat16).eval()
    params = sum(p.numel() for p in model.parameters())
    for rows in args.rows:
        static_in = torch.randn(rows, args.dim, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            # Warm up on a side stream (cuBLAS workspaces, lazy init) before capturing.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    model(static_in)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = model(static_in)

            new = torch.randn_like(static_in)
            static_in.copy_(new)
            graph.replay()
            same = bitwise_equal(static_out, model(new))
            nbytes = params * 2
            eager = B.bench(lambda: model(static_in), name=f"eager rows={rows}", bytes=nbytes, cold_l2=False)
            graphed = B.bench(graph.replay, name=f"cuda graph rows={rows}", bytes=nbytes, cold_l2=False)
        print(f"\n{args.layers} layers, dim {args.dim}, {params / 1e6:.0f}M params, rows={rows}; "
              f"graph output == eager: {same}")
        print(B.table([eager, graphed]))
        print(f"speedup {eager.ms / graphed.ms:.2f}x; overhead removed per step {eager.ms - graphed.ms:.3f} ms")


if __name__ == "__main__":
    main()
