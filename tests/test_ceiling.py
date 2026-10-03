import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "02_decode_ceiling"))

from ceiling import Model, bits_per_weight, decode_ceiling  # noqa: E402


def test_bits_per_weight():
    assert bits_per_weight(4, 64) == 4.5
    assert bits_per_weight(4, 64, has_bias=False) == 4.25
    assert bits_per_weight(16, 64) == 16


def test_dense_27b_on_spark_bandwidth():
    # 27B at 4.5 bits is ~15.2 GB per pass; at 273 GB/s that caps serial decode near 18 tok/s.
    r = decode_ceiling(Model(27, 27), 273)
    assert abs(r["weight_gb"] - 15.19) < 0.01
    assert 17.5 < r["tok_s_per_stream"] < 18.5


def test_speculation_and_streams_scale():
    m = Model(8, 8)
    base = decode_ceiling(m, 273)["tok_s_per_stream"]
    assert abs(decode_ceiling(m, 273, accepted=3)["tok_s_per_stream"] - 3 * base) < 1e-6
    assert abs(decode_ceiling(m, 273, streams=4)["tok_s_total"] - 4 * base) < 1e-6


def test_kv_term():
    m = Model(8, 8, layers=32, kv_heads=8, head_dim=128, kv_bytes=2)
    assert m.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2
    assert decode_ceiling(m, 273, context=32768)["tok_s_per_stream"] < decode_ceiling(m, 273)["tok_s_per_stream"]
