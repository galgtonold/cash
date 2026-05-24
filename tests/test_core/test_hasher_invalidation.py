"""Hasher source hash must participate in the cache key so that
changing a registered hasher's body invalidates dependent cache entries,
even when the new and old hasher produce the same hex output for the
test input.
"""
from __future__ import annotations

import hashlib

import pytest

from cash import Cash


class _Box:
    """Simple wrapper for testing — not a built-in type so it routes
    through the user-registered hasher path, not the built-in path."""
    def __init__(self, v):
        self.v = v


def test_same_hasher_body_preserves_cache_hit(tmp_path):
    """Re-registering the same hasher (identical source) does not
    invalidate cached values."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def hasher(box):
        return hashlib.sha256(str(box.v).encode()).hexdigest()

    c.register_hasher(_Box, hasher)

    n = {"calls": 0}

    @c.cache
    def f(box):
        n["calls"] += 1
        return box.v * 2

    assert f(_Box(5)) == 10
    assert n["calls"] == 1

    # Re-register the SAME function. Source identical → same source hash → cache still hits.
    c.register_hasher(_Box, hasher)
    assert f(_Box(5)) == 10
    assert n["calls"] == 1, (
        f"re-registering an identical hasher should not invalidate the "
        f"cache (calls={n['calls']})"
    )


def test_different_hasher_body_invalidates_even_when_output_matches(tmp_path):
    """The load-bearing test: two hashers with DIFFERENT source bodies
    but IDENTICAL hex output for the test input must produce different
    cache keys."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    # Both hashers return the literal string "same_output" for any input.
    # Their source bodies are deliberately different.
    def hasher_v1(box):
        return "same_output"

    def hasher_v2(box):
        # Different bytecode and different source text from hasher_v1,
        # but the return value is intentionally identical.
        ignored = box.v + 1  # unused work; differentiates the source/bytecode
        del ignored
        return "same_output"

    c.register_hasher(_Box, hasher_v1)

    n = {"calls": 0}

    @c.cache
    def f(box):
        n["calls"] += 1
        return box.v * 3

    r1 = f(_Box(5))
    assert r1 == 15
    assert n["calls"] == 1

    # Re-register with a different-source hasher whose output coincides.
    c.register_hasher(_Box, hasher_v2)

    r2 = f(_Box(5))
    assert r2 == 15
    assert n["calls"] == 2, (
        f"different hasher source must invalidate the cache even when "
        f"the hasher's output is the same (calls={n['calls']})"
    )


def test_hasher_lambda_source_captured(tmp_path):
    """Lambdas defined in this file (where inspect.getsource works) are
    handled by the primary path."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    c.register_hasher(
        _Box,
        lambda box: hashlib.sha256(str(box.v).encode()).hexdigest(),
    )

    n = {"calls": 0}

    @c.cache
    def f(box):
        n["calls"] += 1
        return box.v

    assert f(_Box(7)) == 7
    assert f(_Box(7)) == 7
    assert n["calls"] == 1


def test_hasher_callable_instance_works(tmp_path):
    """Callable objects (instances with `__call__`) work as hashers via
    the `__call__.__code__` fallback. Two instances of the same callable
    class share the same source hash (the bytecode of `__call__`).
    """
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    class HasherImpl:
        def __call__(self, box):
            return hashlib.sha256(str(box.v).encode()).hexdigest()

    c.register_hasher(_Box, HasherImpl())

    n = {"calls": 0}

    @c.cache
    def f(box):
        n["calls"] += 1
        return box.v

    assert f(_Box(11)) == 11
    assert f(_Box(11)) == 11
    assert n["calls"] == 1

    # Replacing with a DIFFERENT instance of the SAME class still hits
    # (the source hash is keyed on the class's __call__ bytecode, not
    # on the instance identity — two instances of HasherImpl produce
    # the same source hash).
    c.register_hasher(_Box, HasherImpl())
    assert f(_Box(11)) == 11
    assert n["calls"] == 1


def test_hasher_exec_defined_function_uses_bytecode_fallback(tmp_path):
    """Functions defined via exec() have no discoverable source via
    inspect.getsource(), but their bytecode (`__code__.co_code`) is
    stable. The fallback path handles this."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    ns = {}
    exec(
        "import hashlib\n"
        "def h(box):\n"
        "    return hashlib.sha256(str(box.v).encode()).hexdigest()\n",
        ns,
    )
    c.register_hasher(_Box, ns["h"])

    n = {"calls": 0}

    @c.cache
    def f(box):
        n["calls"] += 1
        return box.v + 100

    assert f(_Box(2)) == 102
    assert f(_Box(2)) == 102
    assert n["calls"] == 1
