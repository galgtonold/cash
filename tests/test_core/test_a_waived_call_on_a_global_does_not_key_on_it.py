"""A global whose only effectful use is on a waived line is not folded into the key.

Reported against the docmind demo: ``llm.complete`` bills every provider call
with ``LEDGER.record(result)  # @cash:assume-safe``. Every cached function
calling it warned IMPURE-SCOPE-MUTATION for ``LEDGER`` anyway, and a second
pass over the same documents -- in the same process and in a new one -- was
mostly misses: 7 of 8 provider calls again.

``LEDGER.record(...)`` is a method call, so ``LEDGER`` was folded provisionally
and watched. The ledger moved (that is the effect the waiver audits), so the
name was demoted with a warning -- per cached function, after that function's
first call had already keyed on the ledger's count. The waiver says losing that
effect on a hit is fine; keying on it is what made the hits impossible.

A regression from 1f37bb1 (2026-09-09, "reading a global through a method call
still reads it"): before it, any method call on a global kept it out of the key.
98142ef then added a second channel, the instance a bound method carries, so
both have to honour the waiver.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

LLM = textwrap.dedent("""
    import time

    class Ledger:
        def __init__(self):
            self.calls = 0

        def record(self):
            self.calls += 1  # @cash:assume-safe

    LEDGER = Ledger()

    def complete(prompt):
        time.sleep(0.3)  # @cash:assume-safe
        LEDGER.record(){WAIVER}
        return prompt.upper()
""")

MAIN = textwrap.dedent("""
    import cash
    import llm

    @cash.cache
    def extract(doc):
        return llm.complete("extract " + doc)

    @cash.cache
    def summarize(doc):
        return llm.complete("summarize " + doc)

    def run():
        for doc in ["a", "b", "c"]:
            extract(doc)
            summarize(doc)

    run()
    print("FIRST", llm.LEDGER.calls)
    llm.LEDGER.__init__()
    run()
    print("AGAIN", llm.LEDGER.calls)
""")


def _run(tmp_path, waiver):
    (tmp_path / "llm.py").write_text(LLM.replace("{WAIVER}", waiver), encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONWARNINGS="always")
    proc = subprocess.run(
        [sys.executable, "main.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, proc.stderr


def test_a_waived_billing_call_neither_warns_nor_misses(tmp_path):
    out, err = _run(tmp_path, "  # @cash:assume-safe")
    assert "IMPURE-SCOPE-MUTATION" not in err, err
    assert "FIRST 6" in out and "AGAIN 0" in out, out

    out, err = _run(tmp_path, "  # @cash:assume-safe")
    assert "FIRST 0" in out and "AGAIN 0" in out, out


def test_without_the_waiver_it_is_still_reported(tmp_path):
    _, err = _run(tmp_path, "")
    assert "IMPURE-SCOPE-MUTATION" in err, err
