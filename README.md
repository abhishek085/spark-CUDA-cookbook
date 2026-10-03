# spark-CUDA-cookbook

Recipes, tools and methods for writing faster CUDA kernels and LLM inference on an NVIDIA **DGX Spark**
(GB10, sm_121, 128 GB unified memory).

Each recipe is one script you can run, does one job, and checks correctness before it reports speed.
[docs/METHODS.md](docs/METHODS.md) is the playbook that ties them together.

## Quick start (on the Spark)

```bash
PYTORCH_TAG=25.09-py3 scripts/container.sh       # NGC PyTorch container; pick a current tag
python recipes/00_spark_info/spark_info.py        # what this box is
python recipes/01_roofline/measure.py             # measured bandwidth + TFLOP/s -> results/roofline.json
python recipes/02_decode_ceiling/ceiling.py --params 8 --bw-from-results
python recipes/03_int4_gemv/run.py                # 4-bit decode GEMV vs reference and bf16
```

CUDA kernels compile on first use for the GPU in the machine (`common/build.py`), and later runs reuse
the build.

## Recipes

| # | Recipe | What it teaches |
|---|---|---|
| 00 | [spark_info](recipes/00_spark_info/spark_info.py) | Device facts, toolchain, unified-memory headroom |
| 01 | [roofline](recipes/01_roofline/) | Measured DRAM/L2 bandwidth (custom streaming kernels) and bf16/fp16/fp8 matmul peaks |
| 02 | [decode_ceiling](recipes/02_decode_ceiling/ceiling.py) | Upper bounds on tok/s from bytes, context, batching and speculative decoding |
| 03 | [int4_gemv](recipes/03_int4_gemv/) | A 4-bit affine GEMV on MLX's weight layout: naive vs coalesced warp kernel, dequantization folded into the dot product |
| 04 | [fused_rmsnorm](recipes/04_fused_rmsnorm/fused_add_rmsnorm.py) | Triton fusion of residual add + RMSNorm vs eager and `torch.compile` |
| 05 | [cuda_graphs](recipes/05_cuda_graphs/decode_graph.py) | Capturing a decode step to remove launch overhead |
| 06 | [batch_invariance](recipes/06_batch_invariance/check_invariance.py) | Whether a row's bits change with batch size (cuBLAS vs a row-invariant kernel), which exact speculative decoding depends on |
| 07 | [profiling](recipes/07_profiling/) | `nsys` and `ncu` wrappers with what to read in each |

## Layout

```
common/      build.py (JIT for this GPU), bench.py (CUDA-event timing, L2 flush, GB/s),
             verify.py (tolerances, bitwise equality), spark.py (device properties)
recipes/     one directory per technique
docs/        METHODS.md: the optimization playbook
tests/       CPU tests for the pure-Python parts (quantization layout, ceiling math)
scripts/     container.sh
```

## Tests

```bash
python -m pytest -q tests        # runs anywhere, no GPU needed
```

The GPU recipes check themselves: each prints `OK`/`FAIL` against an fp32 reference before
benchmarking, and `03_int4_gemv/run.py` exits non-zero on a failure.

## Status

The CUDA and Triton code hasn't been compiled or run on a Spark yet. The first run on hardware will
show whether it builds and is correct. Measured numbers belong in `results/` (not committed) or in a
recipe's README once they're confirmed.

## Credits

The approach to the decode matmul and the row-invariance rules follow
[TensorFold](https://github.com/ashhart/TensorFold) (Apache 2.0). No TensorFold code is copied here.
