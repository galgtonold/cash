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
