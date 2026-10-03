import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "03_int4_gemv"))

from quant import dequantize, matmul_reference, pack, quantize, unpack  # noqa: E402


def test_pack_unpack_roundtrip():
    q = torch.randint(0, 16, (8, 128))
    assert torch.equal(unpack(pack(q)), q)


def test_nibble_order_is_lowest_first():
    q = torch.zeros(1, 8, dtype=torch.int64)
    q[0, 0], q[0, 7] = 0x1, 0xF
    word = pack(q)[0, 0].item() & 0xFFFFFFFF
    assert word == 0xF0000001


@pytest.mark.parametrize("gs", [32, 64, 128])
def test_quantize_error_is_within_half_a_step(gs):
    torch.manual_seed(0)
    w = torch.randn(16, 256)
    packed, scales, biases = quantize(w, gs)
    err = (dequantize(packed, scales, biases) - w).abs().reshape(16, 256 // gs, gs)
    # bf16-rounded scale/bias add a little on top of the half step
    assert torch.all(err <= scales.float()[..., None] * 0.5 + 1e-2 * w.abs().max())


def test_reference_matches_dense_matmul_of_dequantized():
    torch.manual_seed(0)
    w, x = torch.randn(64, 128), torch.randn(3, 128)
    packed, scales, biases = quantize(w)
    torch.testing.assert_close(matmul_reference(x, packed, scales, biases), x @ dequantize(packed, scales, biases).T)


def test_group_fold_identity():
    # The kernel's rewrite: sum((q*s + b) * x) == s * sum(q*x) + b * sum(x)
    torch.manual_seed(0)
    q, x = torch.randint(0, 16, (64,)).double(), torch.randn(64).double()
    s, b = 0.013, -0.2
    assert torch.isclose(((q * s + b) * x).sum(), s * (q * x).sum() + b * x.sum())
