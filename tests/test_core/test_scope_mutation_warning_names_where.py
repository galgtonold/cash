"""IMPURE-SCOPE-MUTATION names the line that reaches the global, not only the cached function.

Reported against the docmind demo, without the waiver: every cached function
(`extract`, `summarize`, `review`) warned that calling it "modifies the module
global 'LEDGER'". True, but the global lives in ``docmind.llm`` and is moved by
``LEDGER.record(result)`` inside ``llm.complete``, two calls below the function
named. In a larger code base the reader has to trace that by hand, while cash
already knew which helper's read it was watching.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

LLM = textwrap.dedent("""
    class Ledger:
        def __init__(self):
            self.calls = 0

        def record(self):
            self.calls += 1  # @cash:assume-safe

    LEDGER = Ledger()

    def complete(prompt):
        text = prompt.upper()
        LEDGER.record()
        return text
""")

PIPELINE = textwrap.dedent("""
    import llm

    def review(doc):
        return llm.complete("review " + doc)
""")

MAIN = textwrap.dedent("""
    import cash
    import pipeline

    review = cash.cache(pipeline.review)
    review("a")
    review("b")
""")

OWN = textwrap.dedent("""
    import cash

    class Counter:
        def __init__(self):
            self.n = 0

        def bump(self):
            self.n += 1  # @cash:assume-safe

    COUNTER = Counter()

    @cash.cache
    def work(x):
        y = x * 2
        COUNTER.bump()
        return y

    work(1)
    work(2)
""")


def _run(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONWARNINGS="always")
    proc = subprocess.run(
        [sys.executable, "main.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stderr


def _warning(err):
    found = re.search(r"\[IMPURE-SCOPE-MUTATION\][^\n]*", err)
    assert found, err
    return found.group(0)


def test_a_global_moved_in_a_helper_names_the_helper_and_its_line(tmp_path):
    msg = _warning(_run(tmp_path, {"llm.py": LLM, "pipeline.py": PIPELINE, "main.py": MAIN}))
    assert "llm.LEDGER" in msg, msg
    assert "llm.complete" in msg, msg
    assert "LEDGER.record()" in msg, msg
    assert re.search(r"llm\.py:13\b", msg), msg


def test_a_global_moved_in_the_cached_function_names_the_line(tmp_path):
    msg = _warning(_run(tmp_path, {"main.py": OWN}))
    assert "COUNTER.bump()" in msg, msg
    assert re.search(r"main\.py:16\b", msg), msg
