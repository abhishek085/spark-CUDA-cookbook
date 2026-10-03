import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipes" / "03_int4_gemv"))
sys.path.insert(0, str(ROOT / "recipes" / "08_int4_tensorcore"))

from quant import dequantize, matmul_reference, quantize, unpack  # noqa: E402
from tc_pack import pack, pack_codes, pack_fused, split_k, unpack_codes  # noqa: E402


def test_pack_unpack_codes_roundtrip():
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (16, 256))
    assert torch.equal(unpack_codes(pack_codes(codes)), codes)


def test_lane_gets_the_mma_b_fragment():
    # Lane 5 (g=1, t=1), first chunk, step 0: register 0 must hold column 1 at k=2,3; register 1 at k=10,11.
    codes = torch.arange(8 * 128).reshape(8, 128) % 16
    w = pack_codes(codes)[0, 0, 5]                       # (4,) words for lane 5
    word = w[0].item() & 0xFFFFFFFF
    reg0 = (word & 0xF, (word >> 16) & 0xF)              # pair 0
    reg1 = ((word >> 4) & 0xF, (word >> 20) & 0xF)       # pair 1
    assert reg0 == (codes[1, 2].item(), codes[1, 3].item())
    assert reg1 == (codes[1, 10].item(), codes[1, 11].item())


@pytest.mark.parametrize("gs", [32, 64, 128])
def test_pack_pads_and_preserves_weights(gs):
    torch.manual_seed(0)
    w = torch.randn(40, 256)                              # N=40 pads to 64
    packed, scales, biases = quantize(w, gs)
    tw = pack(packed, scales, biases)
    assert tw.npad == 64 and tw.n == 40 and tw.gs == gs
    assert tw.scales.shape == (256 // gs, 64)
    codes = unpack_codes(tw.words)
    assert torch.equal(codes[:40], unpack(packed))
    assert torch.all(codes[40:] == 0) and torch.all(tw.scales[:, 40:] == 0)


def emulate_kernel(x: torch.Tensor, tw) -> torch.Tensor:
    """The kernel's arithmetic in fp64: per group d = sum(q * x), acc += d * s + sum(x) * b."""

    codes = unpack_codes(tw.words)[:tw.n].double()                 # (N, K)
    m, k = x.shape
    kg = k // tw.gs
    xg = x.double().reshape(m, kg, tw.gs)
    qg = codes.reshape(tw.n, kg, tw.gs)
    d = torch.einsum("mgk,ngk->mng", xg, qg)
    s = tw.scales[:, :tw.n].double().T                             # (N, KG)
    b = tw.biases[:, :tw.n].double().T
    return (d * s + xg.sum(-1)[:, None, :] * b).sum(-1)


def test_kernel_math_matches_reference():
    torch.manual_seed(0)
    w, x = torch.randn(48, 384) * 0.02, torch.randn(3, 384)
    packed, scales, biases = quantize(w)
    tw = pack(packed, scales, biases)
    torch.testing.assert_close(emulate_kernel(x, tw).float(), matmul_reference(x, packed, scales, biases),
                               atol=1e-4, rtol=1e-4)


def test_pack_fused_is_concatenation():
    torch.manual_seed(0)
    parts = [quantize(torch.randn(n, 256)) for n in (32, 8, 8)]
    tw = pack_fused(parts)
    assert tw.splits == (32, 8, 8) and tw.n == 48
    full = torch.cat([dequantize(*p) for p in parts])
    x = torch.randn(2, 256)
    torch.testing.assert_close(emulate_kernel(x, tw).float(), x @ full.T, atol=1e-4, rtol=1e-4)


def test_split_k_depends_on_shape_only():
    assert split_k(4096, 4096) == 4          # 128 column groups -> 4 slices reach the 512-warp target
    assert split_k(128256, 4096) == 1        # vocab head already has plenty of work
    assert split_k(4096, 14336) == 4
    for npad, k in [(4096, 4096), (1024, 2048), (14336, 4096)]:
        sk = split_k(npad, k)
        assert (k // 128) % sk == 0
