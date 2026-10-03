# spark-CUDA-cookbook

Recipes, tools and methods for writing faster CUDA kernels and LLM inference on an NVIDIA **DGX Spark**
(GB10, sm_121, 128 GB unified memory).

A [Nokast](https://nokast.substack.com/) open-source project. Nokast builds local-first AI in public:
small and local LLMs, on-device inference, and the build logs, mistakes and trade-offs behind them.
This cookbook is where the low-level performance work lives.

Each recipe is one script you can run, does one job, and checks correctness before it reports speed.
[docs/METHODS.md](docs/METHODS.md) is the playbook that ties them together.

## Quick start (on the Spark)

```bash
git clone https://github.com/abhishek085/spark-CUDA-cookbook.git
cd spark-CUDA-cookbook
PYTORCH_TAG=<current NGC tag> scripts/container.sh   # shell in NVIDIA's PyTorch container, repo at /work

# inside the container:
python recipes/00_spark_info/spark_info.py            # what this box is
python recipes/01_roofline/measure.py                 # measured bandwidth + TFLOP/s -> results/roofline.json
python recipes/02_decode_ceiling/ceiling.py --params 8 --bw-from-results
python recipes/03_int4_gemv/run.py                    # 4-bit decode GEMV vs reference and bf16
python recipes/08_int4_tensorcore/run.py              # tensor-core 4-bit matmul: check, invariance, speed, fused QKV
```

Pick the container tag from the [NGC PyTorch catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
CUDA kernels compile on first use for the GPU in the machine only (`common/build.py`). Builds are cached
in `.cache/`, so later runs and container restarts reuse them.

Run recipe 01 before the others. Recipes 02 and 03 read its `results/roofline.json` to compare against
your machine's measured bandwidth instead of the spec sheet's.

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
| 08 | [int4_tensorcore](recipes/08_int4_tensorcore/) | The faster 4-bit kernel: weights repacked at load into tensor-core order, `mma.sync` over 16-row tiles (decode and draft-verify windows), shape-only split-K, fused Q/K/V in one launch, row-invariant by design |

Example: what is the most an 8B model at 4-bit could decode on this box, alone and with speculative
decoding accepting 3 tokens per pass?

```bash
python recipes/02_decode_ceiling/ceiling.py --params 8 --bw-from-results --accepted 3 --draft-cost 0.1
```

Profiling a kernel:

```bash
recipes/07_profiling/nsys.sh python recipes/05_cuda_graphs/decode_graph.py --rows 1
recipes/07_profiling/ncu.sh gemv_warp python recipes/03_int4_gemv/run.py --shape 14336x4096 --rows 1 --skip-naive
recipes/07_profiling/ncu.sh qmv_tc python recipes/08_int4_tensorcore/run.py --only speed --shape 14336x4096 --rows 1
```

## Layout

```
common/      build.py (JIT for this GPU), bench.py (CUDA-event timing, L2 flush, GB/s),
             verify.py (tolerances, bitwise equality), spark.py (device properties)
recipes/     one directory per technique
docs/        METHODS.md: the optimization playbook
tests/       CPU tests for the pure-Python parts (quantization and tensor-core layouts, ceiling math)
scripts/     container.sh: NGC PyTorch container with the repo mounted
results/     measurements and profiles written by the recipes (not committed)
```

## Tests

```bash
python -m pytest -q tests        # runs anywhere, no GPU needed
```

The kernel recipes check themselves on the GPU. Recipes 03, 04 and 08 print `OK`/`FAIL` against a
reference before benchmarking, and `03_int4_gemv/run.py` and `08_int4_tensorcore/run.py` exit non-zero
on a failure. Recipe 08 also fails if any row's bits change with batch size.

The CPU tests cover recipe 08's weight layout too. They decode the packed words the same way the kernel
does and check that each lane gets the tensor-core fragment it expects, so layout bugs show up before
anything runs on the GPU. Recipe 05 checks
the graph's output against eager, and recipe 06 reports `same`/`DIFF` per batch size.

## Status

The CUDA and Triton code hasn't been compiled or run on a Spark yet. The first run on hardware will
show whether it builds and is correct. Measured numbers belong in `results/` (not committed) or in a
recipe's README once they're confirmed.

## Roadmap

- Recipe 08 follow-ups once it's measured: stage activations in shared memory, a deeper
  `cp.async` pipeline, and a tensor-core prefill path for hundreds of rows
- NVFP4 and FP8 recipes for Blackwell tensor cores
- Speculative decoding: n-gram drafts from the context, and a row-invariant verify pass
- Measured results from a DGX Spark for every recipe

## Follow along

Build logs, write-ups and results from this cookbook are posted on
[Nokast on Substack](https://nokast.substack.com/) and [nokast.com](https://www.nokast.com/).
Issues and pull requests are welcome, especially measurements from your own Spark.

## Credits

The approach to the decode matmul and the row-invariance rules follow
[TensorFold](https://github.com/ashhart/TensorFold) (Apache 2.0). No TensorFold code is copied here.

## License

Apache 2.0, see [LICENSE](LICENSE).
