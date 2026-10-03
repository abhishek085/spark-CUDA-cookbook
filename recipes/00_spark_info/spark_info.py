"""Print what this machine is: GPU, compute capability, memory, toolchain. Run this first on a new box or container.

    python recipes/00_spark_info/spark_info.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.spark import device_summary  # noqa: E402


def run(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return f"({cmd[0]} not found)"
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible to PyTorch")
    info = device_summary()
    print(json.dumps(info, indent=2))
    print("\nnvcc:", run(["nvcc", "--version"]).splitlines()[-1:])
    print("driver:", run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                           "--format=csv,noheader"]))
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch and len(arch.split()) > 1:
        print(f"\nnote: TORCH_CUDA_ARCH_LIST={arch!r} compiles every kernel for each arch; "
              "common/build.py narrows it to this GPU")
    if info["compute_capability"] != "12.1":
        print(f"\nnote: expected 12.1 (GB10) on a DGX Spark, got {info['compute_capability']}")
    # GB10 shares one LPDDR5x pool between CPU and GPU: what the OS page cache and other processes hold
    # is memory your kernels and KV cache cannot use.
    free, total = torch.cuda.mem_get_info()
    print(f"\nGPU-visible memory: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB "
          "(unified with the CPU; drop the page cache before big runs if this looks low)")


if __name__ == "__main__":
    main()
