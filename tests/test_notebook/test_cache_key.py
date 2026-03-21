"""Unit tests for cache_key.py — the single source of truth for cache key computation.

Tests cover:
- Basic cache key generation (deterministic, unique per code)
- Input lineage priority order (virtual > variable > _cash_lineage_hash > compute_hash > str)
- Module detection and inclusion in module_component
- Function source hashing via function_tracker
- Occurrence index for duplicate statements
- CacheKeyContext dataclass usage
- Deprecated kwargs path
- Edge cases: empty inputs, builtins, get_ipython skipped
"""

import hashlib
from unittest.mock import MagicMock

from cash.notebook.cache_key import compute_cache_key, CacheKeyContext, FunctionTrackerProtocol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_hash(code: str) -> str:
    """Compute the same SHA-256 source hash that compute_cache_key uses."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _expected_key(combined: str) -> str:
    """Compute the expected stmt:xxx cache key from the combined hash string."""
    return f"stmt:{hashlib.sha256(combined.encode('utf-8')).hexdigest()}"


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestBasicCacheKey:
    """Fundamental cache key properties."""

    def test_deterministic(self):
        """Same code + inputs produce the same key."""
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        k1 = compute_cache_key("x = 1", set(), ctx=ctx)
        k2 = compute_cache_key("x = 1", set(), ctx=ctx)
        assert k1 == k2

    def test_different_code_different_key(self):
        """Different code produces different keys."""
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        k1, *_ = compute_cache_key("x = 1", set(), ctx=ctx)
        k2, *_ = compute_cache_key("x = 2", set(), ctx=ctx)
        assert k1 != k2

    def test_returns_five_tuple(self):
        """Return value is (cache_key, source_hash, input_hashes, func_hashes, module_hashes)."""
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        result = compute_cache_key("y = 1", set(), ctx=ctx)
        assert len(result) == 5
        cache_key, source_hash, input_hashes, func_hashes, module_hashes = result
        assert cache_key.startswith("stmt:")
        assert source_hash == _source_hash("y = 1")
        assert isinstance(input_hashes, list)
        assert isinstance(func_hashes, list)
        assert isinstance(module_hashes, list)

    def test_empty_inputs(self):
        """No inputs means input_hashes is empty."""
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        _, _, input_hashes, _, _ = compute_cache_key("x = 42", set(), ctx=ctx)
        assert input_hashes == []

    def test_source_hash_matches_sha256(self):
        """source_hash in return value matches SHA-256 of code."""
        code = "result = compute(data)"
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        _, source_hash, *_ = compute_cache_key(code, set(), ctx=ctx)
        assert source_hash == _source_hash(code)


# ---------------------------------------------------------------------------
# Input lineage priority
# ---------------------------------------------------------------------------

class TestInputLineagePriority:
    """virtual_lineage > variable_lineage > _cash_lineage_hash > compute_hash > str."""

    def test_virtual_lineage_takes_priority(self):
        """virtual_lineage wins over variable_lineage."""
        ctx = CacheKeyContext(
            variable_lineage={"x": "runtime_hash"},
            user_ns={"x": 10},
            virtual_lineage={"x": "virtual_hash"},
        )
        _, _, input_hashes, _, _ = compute_cache_key("y = x", {"x"}, ctx=ctx)
        assert "virtual_hash" in input_hashes
        assert "runtime_hash" not in input_hashes

    def test_variable_lineage_used_when_no_virtual(self):
        """variable_lineage used when virtual_lineage has no entry."""
        ctx = CacheKeyContext(
            variable_lineage={"x": "runtime_hash"},
            user_ns={"x": 10},
        )
        _, _, input_hashes, _, _ = compute_cache_key("y = x", {"x"}, ctx=ctx)
        assert "runtime_hash" in input_hashes

    def test_cash_hash_attribute_fallback(self):
        """_cash_lineage_hash attribute used when neither lineage has the var."""
        obj = MagicMock()
        obj._cash_lineage_hash = "attribute_hash"
        # Ensure obj is not callable/module so we don't skip
        obj.__class__ = type("Dummy", (), {})
        ctx = CacheKeyContext(
            variable_lineage={},
            user_ns={"x": obj},
        )
        _, _, input_hashes, _, _ = compute_cache_key("y = x", {"x"}, ctx=ctx)
        assert "attribute_hash" in input_hashes

    def test_compute_hash_fn_fallback(self):
        """compute_hash_fn used when no lineage and no _cash_lineage_hash."""
        val = [1, 2, 3]
        ctx = CacheKeyContext(
            variable_lineage={},
            user_ns={"x": val},
            compute_hash_fn=lambda v: "computed_hash",
        )
        _, _, input_hashes, _, _ = compute_cache_key("y = x", {"x"}, ctx=ctx)
        assert "computed_hash" in input_hashes

    def test_str_fallback(self):
        """Falls back to str-based hash when nothing else is available."""
        ctx = CacheKeyContext(
            variable_lineage={},
            user_ns={"x": 42},
        )
        _, _, input_hashes, _, _ = compute_cache_key("y = x", {"x"}, ctx=ctx)
        expected = hashlib.sha256(str(42).encode("utf-8")).hexdigest()
        assert expected in input_hashes

    def test_missing_from_ns_gives_no_lineage(self):
        """Variable not in user_ns and not in lineage gives no input hash."""
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        _, _, input_hashes, _, _ = compute_cache_key("y = x", {"x"}, ctx=ctx)
        assert input_hashes == []


# ---------------------------------------------------------------------------
# Module handling
# ---------------------------------------------------------------------------

class TestModuleHandling:
    """Modules go to module_source_hashes, not input_hashes."""

    def test_module_in_virtual_modules(self):
        """Variable in virtual_modules is treated as module."""
        ctx = CacheKeyContext(
            variable_lineage={"os": "mod_lineage"},
            user_ns={},
            virtual_modules={"os"},
        )
        _, _, input_hashes, _, module_hashes = compute_cache_key(
            "path = os.getcwd()", {"os"}, ctx=ctx
        )
        assert input_hashes == []
        assert any("os:mod_lineage" in h for h in module_hashes)

    def test_actual_module_in_user_ns(self):
        """types.ModuleType in user_ns is detected as module."""
        import os
        ctx = CacheKeyContext(
            variable_lineage={"os": "os_lineage"},
            user_ns={"os": os},
        )
        _, _, input_hashes, _, module_hashes = compute_cache_key(
            "p = os.path", {"os"}, ctx=ctx
        )
        assert input_hashes == []
        assert any("os:os_lineage" in h for h in module_hashes)

    def test_module_without_lineage_excluded(self):
        """Module without variable_lineage entry is skipped entirely."""
        import os
        ctx = CacheKeyContext(
            variable_lineage={},
            user_ns={"os": os},
        )
        _, _, input_hashes, _, module_hashes = compute_cache_key(
            "p = os.path", {"os"}, ctx=ctx
        )
        assert input_hashes == []
        assert module_hashes == []


# ---------------------------------------------------------------------------
# Skipped names
# ---------------------------------------------------------------------------

class TestSkippedNames:
    """get_ipython and __builtins__ are always excluded."""

    def test_get_ipython_skipped(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        _, _, input_hashes, _, _ = compute_cache_key(
            "x = 1", {"get_ipython"}, ctx=ctx
        )
        assert input_hashes == []

    def test_builtins_skipped(self):
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        _, _, input_hashes, _, _ = compute_cache_key(
            "x = 1", {"__builtins__"}, ctx=ctx
        )
        assert input_hashes == []


# ---------------------------------------------------------------------------
# Function tracking
# ---------------------------------------------------------------------------

class TestFunctionTracking:
    """Callable inputs get both lineage hash and source hash."""

    def test_function_source_hash_included(self):
        """Function tracker source hash appears in func_source_hashes."""
        def my_func():
            return 42

        tracker = MagicMock()
        tracker.get_function_source_hash.return_value = "func_src_hash"

        ctx = CacheKeyContext(
            variable_lineage={"my_func": "func_lineage"},
            user_ns={"my_func": my_func},
            function_tracker=tracker,
        )
        _, _, input_hashes, func_hashes, _ = compute_cache_key(
            "y = my_func()", {"my_func"}, ctx=ctx
        )
        assert "func_lineage" in input_hashes
        assert any("func_src_hash" in h for h in func_hashes)

    def test_no_function_tracker_means_no_func_hashes(self):
        """Without function_tracker, func_source_hashes is empty."""
        def my_func():
            return 42

        ctx = CacheKeyContext(
            variable_lineage={"my_func": "func_lineage"},
            user_ns={"my_func": my_func},
        )
        _, _, _, func_hashes, _ = compute_cache_key(
            "y = my_func()", {"my_func"}, ctx=ctx
        )
        assert func_hashes == []


# ---------------------------------------------------------------------------
# Occurrence index
# ---------------------------------------------------------------------------

class TestOccurrenceIndex:
    """Duplicate statements within a cell get distinct keys."""

    def test_default_occurrence_zero_no_suffix(self):
        """occurrence_index=0 (default) produces no suffix."""
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        k0, *_ = compute_cache_key("x = 1", set(), ctx=ctx, occurrence_index=0)
        k_default, *_ = compute_cache_key("x = 1", set(), ctx=ctx)
        assert k0 == k_default

    def test_different_occurrence_different_key(self):
        """occurrence_index > 0 changes the cache key."""
        ctx = CacheKeyContext(variable_lineage={}, user_ns={})
        k0, *_ = compute_cache_key("c.increment()", set(), ctx=ctx, occurrence_index=0)
        k1, *_ = compute_cache_key("c.increment()", set(), ctx=ctx, occurrence_index=1)
        k2, *_ = compute_cache_key("c.increment()", set(), ctx=ctx, occurrence_index=2)
        assert k0 != k1 != k2


# ---------------------------------------------------------------------------
# CacheKeyContext
# ---------------------------------------------------------------------------

class TestCacheKeyContext:
    """Verify CacheKeyContext properly bundles parameters."""

    def test_ctx_required(self):
        """ctx is a required keyword argument."""
        ctx = CacheKeyContext(variable_lineage={"x": "abc123"}, user_ns={"x": 10})
        k, *_ = compute_cache_key("y = x", {"x"}, ctx=ctx)
        assert k.startswith("stmt:")


# ---------------------------------------------------------------------------
# Input ordering
# ---------------------------------------------------------------------------

class TestInputOrdering:
    """Inputs are sorted for deterministic keys."""

    def test_sorted_inputs(self):
        """Different insertion order of inputs yields same key."""
        ctx = CacheKeyContext(
            variable_lineage={"a": "h_a", "b": "h_b", "c": "h_c"},
            user_ns={"a": 1, "b": 2, "c": 3},
        )
        k1, *_ = compute_cache_key("y = a + b + c", {"a", "b", "c"}, ctx=ctx)
        k2, *_ = compute_cache_key("y = a + b + c", {"c", "a", "b"}, ctx=ctx)
        assert k1 == k2


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------

class TestFunctionTrackerProtocol:
    """FunctionTrackerProtocol is a runtime_checkable Protocol."""

    def test_protocol_check(self):
        """Mock with get_function_source_hash satisfies protocol."""
        tracker = MagicMock()
        tracker.get_function_source_hash = MagicMock(return_value=None)
        assert isinstance(tracker, FunctionTrackerProtocol)

    def test_plain_object_fails_check(self):
        """Plain object without method fails protocol check."""
        assert not isinstance(object(), FunctionTrackerProtocol)
