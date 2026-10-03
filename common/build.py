"""JIT-compile a CUDA extension for the GPU in this machine only.

NVIDIA's PyTorch containers set TORCH_CUDA_ARCH_LIST to every architecture back to sm_80, so a plain
``torch.utils.cpp_extension.load`` compiles each kernel many times. On a DGX Spark (GB10, sm_121) we only
want sm_121, which makes builds several times faster and lets ``-lineinfo`` map Nsight Compute results
back to source lines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence


def arch() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: run this on the DGX Spark (or inside its container)")
    return torch.cuda.get_device_capability()


def arch_flags(arch_specific: bool = False) -> list[str]:
    """nvcc flags for this GPU alone; ``arch_specific`` targets ``sm_XYa`` (needed for some Blackwell-only PTX)."""

    major, minor = arch()
    a = "a" if arch_specific else ""
    return [f"-gencode=arch=compute_{major}{minor}{a},code=sm_{major}{minor}{a}"]


def load(name: str, sources: Sequence[str | Path], *, arch_specific: bool = False,
         extra_cuda_cflags: Sequence[str] = (), verbose: bool = False, **kwargs: Any):
    """``cpp_extension.load`` with this GPU's arch, -O3 and -lineinfo; rebuilds only when a source changes."""

    from torch.utils import cpp_extension

    major, minor = arch()
    # Stops cpp_extension adding its own -gencode list on top of ours.
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    flags = ["-O3", "-lineinfo", "-std=c++17", *arch_flags(arch_specific), *extra_cuda_cflags]
    return cpp_extension.load(name=name, sources=[str(s) for s in sources], extra_cuda_cflags=flags,
                              extra_cflags=["-O3", "-std=c++17"], verbose=verbose, **kwargs)
