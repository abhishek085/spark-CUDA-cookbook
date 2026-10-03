"""What the GPU says about itself. Read these instead of hard-coding Spark numbers."""

from __future__ import annotations


def device_summary(device: int = 0) -> dict:
    import torch

    p = torch.cuda.get_device_properties(device)
    return {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "memory_gib": round(p.total_memory / 2**30, 1),
        "l2_mib": round(getattr(p, "L2_cache_size", 0) / 2**20, 1),
        "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", None),
        "smem_per_block_optin_kib": getattr(p, "shared_memory_per_block_optin", 0) // 1024 or None,
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
