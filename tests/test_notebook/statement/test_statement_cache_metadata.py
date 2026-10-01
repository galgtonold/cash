import pytest

from cash.notebook.statement import StatementCacheMetadata


class TestStatementCacheMetadataRoundTrip:
    def test_to_dict_from_dict_round_trips_declared_fields(self):
        meta = StatementCacheMetadata(
            key="cell:abc",
            timestamp=123.0,
            execution_time=1.5,
            inputs=["a", "b"],
            outputs=["c"],
            source_hash="deadbeef",
            code="c = a + b",
            file_dependencies={"data.csv": {"mtime": 1.0, "size": 10}},
            output_lineages={"c": "lineagehash"},
        )
        restored = StatementCacheMetadata.from_dict(meta.to_dict())
        assert restored.key == "cell:abc"
        assert restored.timestamp == 123.0
        assert restored.execution_time == 1.5
        assert restored.inputs == ["a", "b"]
        assert restored.outputs == ["c"]
        assert restored.source_hash == "deadbeef"
        assert restored.code == "c = a + b"
        assert restored.file_dependencies == {"data.csv": {"mtime": 1.0, "size": 10}}
        assert restored.output_lineages == {"c": "lineagehash"}


class TestStatementCacheMetadataCompat:
    def test_from_dict_ignores_unknown_keys(self):
        # Legacy / backend-private keys (e.g. cell_code, cell_hash) must not blow up.
        meta = StatementCacheMetadata.from_dict(
            {"key": "k", "cell_code": "x = 1", "cell_hash": "abc", "future_field": 42}
        )
        assert meta.key == "k"
        assert not hasattr(meta, "cell_code")
        assert not hasattr(meta, "future_field")

    def test_from_dict_defaults_missing_keys_to_none(self):
        meta = StatementCacheMetadata.from_dict({"key": "k"})
        assert meta.ttl is None
        assert meta.file_dependencies is None
        assert meta.output_lineages is None
        assert meta.skipped_reason is None

    def test_to_dict_omits_none_fields(self):
        d = StatementCacheMetadata(key="k", metadata_only=True).to_dict()
        assert d == {"key": "k", "metadata_only": True}

    def test_is_frozen(self):
        import dataclasses

        meta = StatementCacheMetadata(key="k")
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.key = "other"
