"""Tests for the CashWarning hierarchy and _warn_once dedup."""
from __future__ import annotations

import warnings

import pytest

from cash import (
    Cash,
    CashWarning,
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
)


def test_warning_classes_importable():
    """The three warning classes must be importable from the top-level `cash` package."""
    assert issubclass(CashWarning, UserWarning)
    assert issubclass(CashCacheIneffectiveWarning, CashWarning)
    assert issubclass(CashCacheStoreFailedWarning, CashWarning)


def test_warn_once_dedupes_per_func_and_arg_type(tmp_path):
    """`_warn_once` emits each (category, func_name, arg_type) at most once."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")  # bypass Python's built-in dedup
        c._warn_once(CashCacheIneffectiveWarning, "f", "MyType", "msg one")
        c._warn_once(CashCacheIneffectiveWarning, "f", "MyType", "msg one")  # dup
        c._warn_once(CashCacheIneffectiveWarning, "g", "MyType", "msg one")  # new func
        c._warn_once(CashCacheIneffectiveWarning, "f", "OtherType", "msg one")  # new type
        c._warn_once(CashCacheStoreFailedWarning, "f", "MyType", "msg one")  # new category

    # Expect 4 unique emissions
    assert len(captured) == 4, [str(w.message) for w in captured]


def test_warn_once_does_not_emit_when_already_seen(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for _ in range(10):
            c._warn_once(CashCacheIneffectiveWarning, "f", "X", "boom")
    assert len(captured) == 1
    assert "boom" in str(captured[0].message)
    assert captured[0].category is CashCacheIneffectiveWarning
