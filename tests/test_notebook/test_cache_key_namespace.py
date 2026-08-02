"""The key builder must namespace keys without disturbing existing ones."""
import ast

from cash.notebook.cache_key import CacheKeyContext, compute_cache_key


def _ctx():
    return CacheKeyContext(variable_lineage={"x": "deadbeef"}, user_ns={"x": 1})


def test_default_namespace_is_stmt_and_byte_identical():
    result = compute_cache_key("y = x + 1", {"x"}, ctx=_ctx())
    assert result.cache_key.startswith("stmt:")


def test_explicit_call_namespace_changes_only_the_prefix():
    stmt = compute_cache_key("compute(x)", {"x"}, ctx=_ctx())
    call = compute_cache_key("compute(x)", {"x"}, ctx=_ctx(), namespace="call")
    assert call.cache_key == "call:" + stmt.cache_key.removeprefix("stmt:")


def test_namespaces_cannot_collide():
    stmt = compute_cache_key("compute(x)", {"x"}, ctx=_ctx())
    call = compute_cache_key("compute(x)", {"x"}, ctx=_ctx(), namespace="call")
    assert stmt.cache_key != call.cache_key
