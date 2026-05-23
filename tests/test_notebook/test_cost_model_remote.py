"""Cost-model coefficients for Redis + S3 backends.

These are *estimated* values, not benchmarked the way RAM/DISK are.
The tests pin the rough order-of-magnitude shape so the model stays
self-consistent under refactoring; they do NOT pin exact constants.
"""
from __future__ import annotations

import pytest

from cash.notebook import cost_model


# Representative data families for the matrix.
FAMILIES = ["ndarray_dense", "dataframe_numeric", "dict_shallow", "bytes"]


class TestKnownBackends:
    def test_redis_recognised(self):
        # Before this change, anything not in {'ram', 'disk'} silently
        # mapped to 'disk'. Now 'redis' must be a first-class backend.
        assert "redis" in cost_model._KNOWN_BACKENDS

    def test_s3_recognised(self):
        assert "s3" in cost_model._KNOWN_BACKENDS

    def test_unknown_backend_still_falls_back_to_disk(self):
        # Future tiers we haven't modelled (e.g. 'gcs') still degrade
        # gracefully — they're routed to disk coefficients, not crash.
        t = cost_model.estimated_serialize_time("DataFrame", 1024, "gcs")
        t_disk = cost_model.estimated_serialize_time("DataFrame", 1024, "disk")
        assert t == t_disk


class TestRedisShape:
    """Redis: LAN-bound, conservative ~50 MB/s, ~500 us round-trip."""

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("op", ["serialize", "deserialize"])
    def test_coefficients_present(self, family, op):
        assert (family, "redis", op) in cost_model._COEFFS

    def test_small_object_dominated_by_latency(self):
        # 1 KB object: should be in the sub-millisecond range, not in disk's ~10 ms range.
        t = cost_model.estimated_serialize_time("DataFrame", 1024, "redis")
        assert 1e-4 < t < 5e-3, f"1 KB Redis serialize was {t*1000:.2f} ms — expected 0.1–5 ms"

    def test_large_object_dominated_by_bandwidth(self):
        # 100 MB object on Redis at ~50 MB/s should land in the 1-3 s range.
        t = cost_model.estimated_serialize_time("DataFrame", 100 * 1024 * 1024, "redis")
        assert 1.0 < t < 5.0, f"100 MB Redis serialize was {t:.2f} s — expected 1–5 s"

    def test_redis_slower_than_ram_faster_than_s3(self):
        size = 10 * 1024 * 1024  # 10 MB
        ram = cost_model.estimated_serialize_time("DataFrame", size, "ram")
        redis = cost_model.estimated_serialize_time("DataFrame", size, "redis")
        s3 = cost_model.estimated_serialize_time("DataFrame", size, "s3")
        assert ram < redis < s3, f"ordering wrong: ram={ram*1000:.1f}ms redis={redis*1000:.1f}ms s3={s3*1000:.1f}ms"


class TestS3Shape:
    """S3: same-region, conservative ~20 MB/s, ~80 ms request setup."""

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("op", ["serialize", "deserialize"])
    def test_coefficients_present(self, family, op):
        assert (family, "s3", op) in cost_model._COEFFS

    def test_small_object_dominated_by_request_latency(self):
        # S3's per-request overhead dwarfs 1 KB transfer. Predict >= 30 ms.
        t = cost_model.estimated_serialize_time("DataFrame", 1024, "s3")
        assert t >= 0.03, f"1 KB S3 serialize was {t*1000:.2f} ms — expected ≥ 30 ms"
        assert t < 0.5, f"1 KB S3 serialize was {t*1000:.2f} ms — expected < 500 ms"

    def test_large_object_dominated_by_bandwidth(self):
        # 100 MB on S3 at ~20 MB/s ≈ 5 s
        t = cost_model.estimated_serialize_time("DataFrame", 100 * 1024 * 1024, "s3")
        assert 3.0 < t < 10.0, f"100 MB S3 serialize was {t:.2f} s — expected 3–10 s"


class TestGenericFallback:
    """An unknown family + redis/s3 should still produce a number (using _GENERIC)."""

    def test_unknown_family_on_redis(self):
        t = cost_model.estimated_serialize_time("MyCustomType", 1024, "redis")
        assert t > 0  # has *some* prediction

    def test_unknown_family_on_s3(self):
        t = cost_model.estimated_serialize_time("MyCustomType", 1024, "s3")
        assert t > 0
