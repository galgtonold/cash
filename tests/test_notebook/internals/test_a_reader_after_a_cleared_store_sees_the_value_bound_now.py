"""A statement whose key cannot see the code it runs (``inner.__code__``
swapped) is checked against its entry's trace and, when that code changed,
keyed by the value it binds. With the entry gone (the store cleared, or the
rerun not stored again) there was nothing to compare with: the statement
kept the lineage of its key, and a reader below was served the value built
on an earlier run under the same key. The trace kept from its last store
still says the code changed."""

from __future__ import annotations

import sys

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import shadow
from cash.notebook.call_entries import CallEntries

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="the trace runs on 3.12+")
_PERSIST = CacheAnnotation(persist=True)
N = 300_000


@pytest.fixture
def trace_on(monkeypatch):
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    before = shadow.SHADOW
    shadow.disable()
    yield shadow.enable() if shadow.requested() else None
    shadow.disable()
    if before is not None:
        shadow.enable()


@pytest.mark.parametrize("clear", [True, False], ids=["store cleared", "store kept"])
def test_a_reader_after_a_rerun_with_no_entry_sees_the_value_bound_now(
    trace_on, statement_processor, mock_shell, monkeypatch, cash_magics, clear
):
    np = pytest.importorskip("numpy")
    ns = mock_shell.user_ns
    ns["np"] = np
    # A swapped run is not stored again.
    monkeypatch.setattr(CallEntries, "restore_pays", lambda self, result, elapsed: False)

    def run(code):
        statement_processor.process_statement(code, annotation=_PERSIST)

    run("def inner(n):\n    return np.arange(n)")
    run(f"t = inner({N})")
    run("s = int(t.sum())")
    for i, delta in enumerate([1, None, 0, 1, 0]):
        if delta is None:
            if clear:
                cash_magics._cash_instance.backend.clear()
            continue
        swapped = eval(compile(f"lambda n: np.arange(n) + {delta}", f"<v{i}>", "eval"), {"np": np})
        ns["inner"].__code__ = swapped.__code__
        run(f"t = inner({N})")
        run("s = int(t.sum())")
        assert ns["s"] == int((np.arange(N) + delta).sum()), (i, delta)
