"""A miss the trace caught names code picked by name only when there was some.

The trace is kept beside the key for every traced call, as a check on top of
it, not only for a call that picks code by a run-time name. A miss it caught
on a call that picks nothing by name said "code picked by a run-time name
changed", sending the user looking for a ``getattr`` that is not there. It now
says the code the call ran changed; the old words stay for a call that does
pick code by name.
"""

from __future__ import annotations

import sys
import types

import pytest

from cash.decorator.cache_metadata import CacheMetadata
from cash.decorator.explain import MissKind

pytestmark = [pytest.mark.core]


@pytest.fixture
def stale_trace(monkeypatch):
    """Metadata whose trace recorded a global that has moved on since."""
    mod = types.ModuleType("traced_label_mod")
    mod.RATE = 2
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return CacheMetadata(code_trace=[("global", mod.__name__, "RATE", "a digest the value no longer has")])


@pytest.mark.parametrize(("needs_trace", "kind"), [(False, MissKind.RAN), (True, MissKind.TRACE)])
def test_the_miss_names_a_pick_by_name_only_for_a_call_that_made_one(cash_instance, stale_trace, needs_trace, kind):
    verdict = cash_instance._runner.entry_verdict("k", stale_trace, None, quiet=True, needs_trace=needs_trace)
    assert verdict is not None
    assert verdict.kind is kind
    assert "RATE" in verdict.detail
