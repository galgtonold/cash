"""Unit tests for the unified cache key computation module.

Tests ``compute_cache_key`` and ``CacheKeyContext`` directly to verify
cache key determinism, input lineage resolution priority, and module
component inclusion.
"""

import hashlib

from cash.notebook.cache_key import CacheKeyContext, compute_cache_key


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestCacheKeyDeterminism:
    """Cache keys must be reproducible given the same inputs."""

    def test_same_inputs_same_key(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        k1, *_ = compute_cache_key("x = 1", set(), ctx=ctx)
        k2, *_ = compute_cache_key("x = 1", set(), ctx=ctx)
        assert k1 == k2

    def test_different_code_different_key(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        k1, *_ = compute_cache_key("x = 1", set(), ctx=ctx)
        k2, *_ = compute_cache_key("x = 2", set(), ctx=ctx)
        assert k1 != k2

    def test_occurrence_index_changes_key(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        k1, *_ = compute_cache_key("x = 1", set(), ctx=ctx, occurrence_index=0)
        k2, *_ = compute_cache_key("x = 1", set(), ctx=ctx, occurrence_index=1)
        assert k1 != k2

    def test_key_starts_with_stmt_prefix(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        key, *_ = compute_cache_key("x = 1", set(), ctx=ctx)
        assert key.startswith("stmt:")


# ---------------------------------------------------------------------------
# Input lineage priority
# ---------------------------------------------------------------------------

class TestInputLineagePriority:
    """virtual_lineage takes priority over variable_lineage."""

    def test_virtual_lineage_preferred_over_variable_lineage(self):
        virt = {"a": "virtual_hash_a"}
        var = {"a": "variable_hash_a"}
        ctx_virt = CacheKeyContext(
            variable_lineage=var, user_ns={"a": 42}, virtual_lineage=virt
        )
        ctx_var = CacheKeyContext(
            variable_lineage=var, user_ns={"a": 42}, virtual_lineage=None
        )
        k_virt, *_ = compute_cache_key("x = a + 1", {"a"}, ctx=ctx_virt)
        k_var, *_ = compute_cache_key("x = a + 1", {"a"}, ctx=ctx_var)
        # Virtual lineage should produce a different key
        assert k_virt != k_var

    def test_variable_lineage_used_when_no_virtual(self):
        var = {"a": "var_hash_a"}
        ctx = CacheKeyContext(
            variable_lineage=var, user_ns={"a": 42}, virtual_lineage=None
        )
        key, *_ = compute_cache_key("x = a + 1", {"a"}, ctx=ctx)
        assert key.startswith("stmt:")


# ---------------------------------------------------------------------------
# Context vs keyword arguments
# ---------------------------------------------------------------------------

class TestContextRequired:
    """CacheKeyContext is now the only way to pass parameters."""

    def test_ctx_produces_key(self):
        lineage = {"a": "hash_a"}
        ns = {"a": 10}

        k_ctx, *_ = compute_cache_key(
            "x = a",
            {"a"},
            ctx=CacheKeyContext(variable_lineage=lineage, user_ns=ns),
        )
        assert k_ctx.startswith("stmt:")


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    """compute_cache_key returns a 5-tuple."""

    def test_returns_five_elements(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        result = compute_cache_key("x = 1", set(), ctx=ctx)
        assert len(result) == 5

    def test_source_hash_is_sha256(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        _, source_hash, *_ = compute_cache_key("x = 1", set(), ctx=ctx)
        expected = hashlib.sha256("x = 1".encode("utf-8")).hexdigest()
        assert source_hash == expected

    def test_sorted_input_lineages_list(self):
        lineage = {"a": "hash_a", "b": "hash_b"}
        ctx = CacheKeyContext(variable_lineage=lineage, user_ns={"a": 1, "b": 2})
        _, _, sorted_lineages, *_ = compute_cache_key(
            "x = a + b", {"a", "b"}, ctx=ctx
        )
        assert isinstance(sorted_lineages, list)
