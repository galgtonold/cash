"""A cached function that reads a module global which is reassigned/mutated
elsewhere should be flagged (the result would go stale when the global changes).
Reads of never-written globals (constants, lookup tables) must stay quiet, and a
local that merely shares a name with a global must not trigger a false positive.
"""
from __future__ import annotations

import warnings

import pytest

from cash import CashImpurityWarning, CashImpureFunctionError
from cash.purity_analyzer import (
    ISSUE_MUTABLE_GLOBAL,
    _module_modified_globals,
    get_analyzer,
)

from tests.test_core import _global_read_fixture as gf


def _global_flags(func):
    report = get_analyzer().analyze(func.__wrapped__)
    return {
        i.description.split("'")[1]
        for i in report.issues if i.kind == ISSUE_MUTABLE_GLOBAL
    }


def test_scanner_finds_only_modified_globals():
    modified = _module_modified_globals(gf)
    assert "CONFIG" in modified           # in-place mutated (inside a function)
    assert "COUNTER" in modified          # reassigned via `global`
    assert "TABLE" not in modified        # constant (local shadow doesn't count)
    assert "LIMIT" not in modified        # constant int
    assert "REGISTRY" not in modified     # mutated only at import time (top level)


def test_import_time_only_global_not_flagged():
    """A global populated at module top level (import-time init) but never
    mutated inside a function is constant at runtime - reading it is safe."""
    assert _global_flags(gf.from_registry) == set()


def test_reads_mutated_global_is_flagged():
    assert _global_flags(gf.price) == {"CONFIG"}
    assert _global_flags(gf.with_counter) == {"COUNTER"}


def test_reads_constant_global_not_flagged():
    assert _global_flags(gf.lookup) == set()


def test_default_mode_warns_about_the_global():
    gf.price.cache_clear()                # reset warn-once dedup
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gf.price(100)
        msgs = [str(x.message) for x in w if issubclass(x.category, CashImpurityWarning)]
    assert any("CONFIG" in m for m in msgs), msgs


def test_constant_read_does_not_warn():
    gf.lookup.cache_clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gf.lookup("a")
        msgs = [str(x.message) for x in w if issubclass(x.category, CashImpurityWarning)]
    assert not any("TABLE" in m or "LIMIT" in m for m in msgs), msgs


def test_strict_mode_raises_on_mutated_global():
    with pytest.raises(CashImpureFunctionError) as exc:
        gf.strict_price(100)
    assert "CONFIG" in str(exc.value)
