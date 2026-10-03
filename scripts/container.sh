#!/usr/bin/env bash
# Open a shell in NVIDIA's PyTorch container with this repo mounted at /work. The image ships CUDA, nvcc,
# PyTorch, Triton and the Nsight tools built for GB10 (arm64), which is the least painful setup on a Spark.
#
#   scripts/container.sh                 # interactive shell
#   scripts/container.sh python recipes/00_spark_info/spark_info.py
#
# Pick a current tag from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch and pass it as
# PYTORCH_TAG (e.g. PYTORCH_TAG=25.09-py3). Kernel builds are cached in .cache/ so restarts reuse them.
set -euo pipefail
: "${PYTORCH_TAG:?set PYTORCH_TAG to an NGC PyTorch tag, e.g. PYTORCH_TAG=25.09-py3}"
cd "$(dirname "$0")/.."
mkdir -p .cache/torch_extensions .cache/triton
args=(--rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
      --cap-add=SYS_ADMIN                                   # Nsight Compute performance counters
      -v "$PWD:/work" -w /work
      -e TORCH_EXTENSIONS_DIR=/work/.cache/torch_extensions -e TRITON_CACHE_DIR=/work/.cache/triton)
[[ -t 0 ]] && args+=(-it)
exec docker run "${args[@]}" "nvcr.io/nvidia/pytorch:${PYTORCH_TAG}" "${@:-bash}"
