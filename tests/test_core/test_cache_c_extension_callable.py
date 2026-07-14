"""CAS-113: caching a C-extension / dispatcher / builtin callable must not crash.

Before the fix, decorating an opaque callable crashed on decoration or first
call:

* ``c.cache(np.mean)`` — a numpy ``_ArrayFunctionDispatcher`` whose source is
  reachable via ``__wrapped__`` but which has no ``__globals__`` — raised
  ``AttributeError`` in ``find_called_functions``.
* ``c.cache(len)`` / ``c.cache(np.add)`` — a builtin / ufunc with no source and
  no ``__code__`` — raised ``ValueError`` from ``get_source_hash``.
* ``c.cache(functools.partial(...))`` — no ``__name__``/``__qualname__`` — raised
  ``AttributeError`` from ``_get_func_key``.

Opaque callables are now keyed on a stable identity, source/dependency analysis
is skipped for them, and they cache without crashing.
"""
from __future__ import annotations

import functools

import numpy as np

from cash import Cash, FileBackend


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


class _Doubler:
    """Module-level (picklable) class so bound-method self-hashing is clean."""

    def __init__(self, base):
        self.base = base

    def method(self, x):
        return self.base + x


def test_cache_numpy_dispatcher_mean(tmp_path):
    """np.mean: dispatcher with source-via-__wrapped__ but no __globals__."""
    c = _cash(tmp_path)
    f = c.cache(np.mean)
    assert f(np.array([1.0, 2.0, 3.0])) == 2.0
    assert f(np.array([1.0, 2.0, 3.0])) == 2.0  # 2nd identical call -> hit
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


def test_cache_builtin_len(tmp_path):
    """len: pure builtin — no source, no __code__."""
    c = _cash(tmp_path)
    f = c.cache(len)
    assert f([1, 2, 3]) == 3
    assert f([1, 2, 3]) == 3
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


def test_cache_ufunc_add(tmp_path):
    """np.add: numpy ufunc — no source, no __code__."""
    c = _cash(tmp_path)
    f = c.cache(np.add)
    assert f(1, 2) == 3
    assert f(1, 2) == 3
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


def test_cache_functools_partial(tmp_path):
    """functools.partial: no __name__/__qualname__."""
    c = _cash(tmp_path)
    f = c.cache(functools.partial(max, 5))
    assert f(3) == 5
    assert f(3) == 5
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


def test_cache_bound_method(tmp_path):
    """A bound method of a picklable instance caches and hits cleanly."""
    c = _cash(tmp_path)
    f = c.cache(_Doubler(10).method)
    assert f(4) == 14
    assert f(4) == 14
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


def test_distinct_opaque_callables_do_not_collide(tmp_path):
    """Two opaque callables share no source, but their distinct func_name keeps
    their cache entries separate — no cross-function collision."""
    c = _cash(tmp_path)
    f_len = c.cache(len)
    f_add = c.cache(np.add)
    assert f_len([1, 2, 3, 4]) == 4
    assert f_add(1, 2) == 3  # must NOT return len's cached 4
