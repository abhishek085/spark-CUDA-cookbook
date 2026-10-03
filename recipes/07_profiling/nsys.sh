#!/usr/bin/env bash
# Timeline profile with Nsight Systems: where does a whole step's time go (kernels, gaps, launches, syncs)?
# Start here, before Nsight Compute: a 2x kernel speedup is worthless if the GPU is idle between launches.
#
#   recipes/07_profiling/nsys.sh python recipes/05_cuda_graphs/decode_graph.py --rows 1
#   nsys stats results/profiles/<name>.nsys-rep      # or open the .nsys-rep in the Nsight Systems GUI
#
# Look for: gaps between kernels (launch-bound -> CUDA graphs / fusion), cudaStreamSynchronize or
# cudaMemcpy in the decode loop (host syncs), and which kernels top the "cuda_gpu_kern_sum" table.
set -euo pipefail
[[ $# -gt 0 ]] || { sed -n '2,9p' "$0"; exit 1; }
out="results/profiles/nsys_$(date +%Y%m%d_%H%M%S)"
mkdir -p results/profiles
nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --force-overwrite=true -o "$out" "$@"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format table "$out.nsys-rep" | head -60
echo "wrote $out.nsys-rep"
