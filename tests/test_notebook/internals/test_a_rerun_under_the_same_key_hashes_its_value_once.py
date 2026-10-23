"""A statement the trace refuses on every run hashes its new value once.

Run again under the same key (code it ran changed in place), a statement
compares what it bound with what its entry held, to keep the formula's
lineage when they are equal. It hashed both, the stored value read back
included: a 16 MB array was hashed twice per Run All. The entry now keeps
the digest the rerun took, and the next rerun compares with it.
"""

from __future__ import annotations

import sys

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import shadow
from tests._work_counts import hashed_bytes

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the trace that sees code changed in place runs on 3.12+"
)

_PERSIST = CacheAnnotation(persist=True)
MB = 1 << 20


@pytest.fixture
def trace_default(monkeypatch):
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    before = shadow.SHADOW
    shadow.disable()
    yield shadow.enable() if shadow.requested() else None
    shadow.disable()
    if before is not None:
        shadow.enable()


def test_a_rerun_hashes_the_new_value_and_not_the_stored_one(trace_default, statement_processor, mock_shell):
    np = pytest.importorskip("numpy")
    mock_shell.user_ns["np"] = np
    run = lambda code: statement_processor.process_statement(code, annotation=_PERSIST)
    run("def inner(n):\n    return np.arange(n)")
    run("t = inner(2_000_000)")
    array_bytes = 2_000_000 * 8
    counted = []
    for body in ("np.arange(n) + 0", "np.arange(n) * 1", "np.arange(n) - 0"):
        changed = eval(compile(f"lambda n: {body}", "<variant>", "eval"), {})
        mock_shell.user_ns["inner"].__code__ = changed.__code__
        with hashed_bytes() as hashed:
            run("t = inner(2_000_000)")
        counted.append(hashed.bytes)
    # The first rerun reads the stored value back to compare (the entry was
    # stored before any rerun); the later ones compare with the digest kept.
    assert all(count < 1.5 * array_bytes for count in counted[1:]), [c / MB for c in counted]
