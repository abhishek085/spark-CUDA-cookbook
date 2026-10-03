# Methods: making inference faster on a DGX Spark

This is the playbook the recipes implement. Work through it in order. Most wasted effort comes from
optimizing a kernel before checking whether that kernel is what limits the step.

## The loop

1. **Measure this machine.** Run recipe 01. Use your own bandwidth and TFLOP/s numbers, not the
   spec sheet's.
2. **Compute the ceiling.** Run recipe 02. If the plain decode ceiling for your model is 60 tok/s,
   no kernel will get you 200. Only fewer bytes (quantization), fewer passes (speculative decoding)
   or more streams per pass (batching) can.
3. **Profile the whole step.** Use `nsys` (recipe 07). Find the gaps, host syncs and top kernels.
4. **Fix the biggest gap first.** Use a CUDA graph if the GPU is idle between launches, fusion for
   chains of small ops, or a custom kernel for a hot matmul that runs well under the bandwidth line.
5. **Profile that kernel.** Use `ncu` (recipe 07). Check whether it's memory- or compute-bound and
   what it stalls on.
6. **Check correctness every time.** Compare against an fp32 reference. On the verify path, also
   check that each row's bits stay the same at every batch size (recipe 06).
7. **Record what you measured.** Keep the shapes, versions, launch command and numbers next to the
   change.

## Spark facts that change the approach

These are true for GB10 as shipped. Confirm them on your box with recipe 00.

- **Compute capability 12.1 (sm_121).** Build for it alone (`common/build.py`). NGC containers
  otherwise compile every kernel for every architecture back to sm_80.
- **One unified LPDDR5x pool, about 128 GB, shared by CPU and GPU.** No PCIe copies are needed, but
  the OS page cache, Docker and your Python process all take memory the GPU could have used. Budget
  for it, and drop the page cache before big loads. Bandwidth (about 273 GB/s on the spec sheet) is far
  below a discrete HBM GPU's, so **decode is memory-bound even more than usual**, and every byte you
  stop reading pays off directly.
- **Lots of compute per byte.** The ridge point (bf16 TFLOP/s ÷ GB/s, printed by recipe 01) is high.
  Prefill and verifying 16–128 draft rows cost little more than one row, which is why speculative
  decoding pays off so much here.
- **Blackwell tensor cores with FP8 and FP4.** NVFP4 and FP8 weights and activations are the next step
  after int4 GEMV. Some Blackwell-only PTX needs the arch-specific target (`sm_121a`); use
  `load(..., arch_specific=True)`.

## Decode: the memory-bound regime (1–8 rows)

What to do, from the most impact to the least:

1. **Read fewer bytes.** 4-bit with group-64 affine scales is 4.5 bits per weight, 3.6× fewer bytes
   than bf16. Store the KV cache in FP8. On hybrid models, only the attention layers need KV at all.
2. **Lay out weights for the kernel, once, at load time.** Every warp load should be 16 bytes per lane,
   consecutive across lanes (recipe 03's `warp` kernel). TensorFold goes further and repacks weights
   into tensor-core fragment order, so the inner loop does no shuffling.
3. **Fold dequantization into the math.** Use `Σ(q·s + b)·x = s·Σq·x + b·Σx`. The scale and bias are
   applied once per group, never to each weight. You can also compute `Σx` per group in the
   preceding norm kernel and pass it in.
4. **Stream the weights past the cache** with `__ldcs` / `ld.global.cs`. They're used once; the
   activations deserve the cache instead.
5. **Run several projections in one launch.** Q, K and V (or gate and up) read the same input, so one
   kernel can produce all of them.
6. **Remove launch overhead** with CUDA graphs (recipe 05). Capture one graph per row count you serve.
7. **Fuse the glue** (norm, SwiGLU, rotary, gates) in Triton (recipe 04). Fewer launches, fewer
   round trips through memory.

The target is a GEMV that reaches 80–90% of recipe 01's read bandwidth on large shapes. Small shapes
fall short of that because there aren't enough blocks to fill the SMs. Split K across blocks for those,
using a **fixed** split based on the weight's shape.

## Speculative decoding: most of the remaining speedup

A plain 27B model at 4-bit on a Spark tops out around 18 tok/s. TensorFold serves it at 58 tok/s,
roughly 3.2 tokens per pass. The kernels make that possible by keeping a pass over 16–128 rows almost
as cheap as a pass over one. Options for the draft:

- **Draft model** (DFlash2, EAGLE-style heads, a small model from the same family).
- **MTP heads** that the model already ships with.
- **Copying from the context** (n-gram lookup). It costs nothing and works very well on code and
  quoted text.
- **Draft trees** instead of a single chain. They verify more candidates in the same pass.

**Exactness rule.** A draft is accepted only when it equals what serial decoding would produce. That
holds only if row r's result doesn't depend on how many rows ran with it:

- Pick split-K and attention partitions from the weight shape or key range, **never from the row
  count**.
- Use a fixed reduction order. When combining partial results across GPUs, gather them and add in
  rank order instead of using NCCL all-reduce.
- Don't use library GEMMs that switch algorithm by M on an unchecked verify path (recipe 06 shows
  cuBLAS doing this).
- Serial decoding and verification go through the same kernels.
- Where rounding matters, write the FMAs explicitly (or use `--fmad=false`) so the compiler can't
  contract differently in different template instantiations.

## Prefill: the compute-bound regime

At hundreds or thousands of rows the arithmetic dominates. Use a tensor-core GEMM (`mma.sync`, or
`tcgen05` / CUTLASS on Blackwell), FP8 activations with per-row scales (TensorFold reports about 52%
faster prefill), and chunked prefill (for example 4,096 tokens per chunk) to bound memory use.

## Custom-kernel checklist

- [ ] An fp32 PyTorch reference and a test against it
- [ ] Tensor shapes, dtypes and contiguity checked on the host (`TORCH_CHECK`)
- [ ] Launched on `at::cuda::getCurrentCUDAStream()` so CUDA graphs and streams work
- [ ] `C10_CUDA_KERNEL_LAUNCH_CHECK()` after every launch
- [ ] Built with `-lineinfo`, and an `ncu` run that confirms which limit it hits
- [ ] Benchmarked with a cold L2 and warm-up runs, reporting the median and p10/p90
- [ ] Verify-path kernels: a row-invariance check at every row count you serve

## Further reading in code

- [TensorFold](https://github.com/ashhart/TensorFold) (Apache 2.0): `src/tensorfold/cuda/kernels/qmm.cu`
  (4-bit tensor-core matmul on MLX's layout), `gdn.cu` (draft-tree state updates),
  `docs/recipes/adding-a-cuda-family.md` (its exactness rules). This cookbook's approach follows it.
- CUTLASS and CuTe for Blackwell tensor-core GEMMs; the Triton tutorials for fused kernels.
