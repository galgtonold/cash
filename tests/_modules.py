"""Modules a test imports, gone again after it."""

from __future__ import annotations

import sys

import pytest


def forget_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Take *name* out of ``sys.modules`` for the test, and whatever the test
    imports under it out again after.

    ``monkeypatch.delitem(sys.modules, name, raising=False)`` alone undoes
    nothing when the name was not there: the module the test then imported
    stayed loaded for every later test on the worker. A ``wc_memo`` holding
    a ``functools.cache`` memo was found by the helper memo tests that ran
    after it under xdist, and asked about. Set first, the name is restored
    to what it was before, absence included.
    """
    monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, name)
