"""Unit tests for the CacheMetadata edge dataclass.

CacheMetadata is the typed, in-memory view of cache-entry metadata used by
producers/consumers at the cash layer. Backends round-trip a plain dict; the
dataclass converts at the edges via to_dict()/from_dict().
"""

import pytest

from cash.backends import CacheMetadata


class TestCacheMetadataRoundTrip:
    def test_to_dict_from_dict_round_trips_declared_fields(self):
        meta = CacheMetadata(
            key="abc",
            created_at=123.0,
            execution_time=1.5,
            func_name="mod.fn",
        )

        restored = CacheMetadata.from_dict(meta.to_dict())

        assert restored.key == "abc"
        assert restored.created_at == 123.0
        assert restored.execution_time == 1.5
        assert restored.func_name == "mod.fn"


class TestCacheMetadataCompat:
    def test_from_dict_ignores_unknown_keys(self):
        # Backend-private and stale keys (e.g. an older/newer cash version)
        # must not blow up construction.
        meta = CacheMetadata.from_dict(
            {"key": "k", "compressed": True, "some_future_field": 42}
        )

        assert meta.key == "k"
        assert not hasattr(meta, "compressed")
        assert not hasattr(meta, "some_future_field")

    def test_from_dict_defaults_missing_keys_to_none(self):
        meta = CacheMetadata.from_dict({"key": "k"})

        assert meta.ttl is None
        assert meta.storage is None
        assert meta.func_name is None

    def test_to_dict_omits_none_fields(self):
        # Wire format matches the historical "only set keys are present" shape,
        # so backend presence-checks ('x' not in metadata) keep working.
        d = CacheMetadata(key="k", size=10).to_dict()

        assert d == {"key": "k", "size": 10}

    def test_is_frozen(self):
        import dataclasses

        meta = CacheMetadata(key="k")
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.key = "other"  # type: ignore[misc]
