"""``df = etl.clean(raw)`` with a ``@cash.cache`` ``clean`` reads ``raw`` once
or twice, not three times.

The statement's in-place-change fingerprint read every byte of ``raw``
before it ran, the decorator read them again for its key, and the
fingerprint read them once more after: 3 x 0.4-1.1 s for a 4.2M-row polars
frame around a 1.2 s body. When the statement is nothing but that call,
only names are read between the fingerprint and the key, so the decorator
takes the fingerprint's digest (`cash.handed_digests`); and when the call is
served, its body never runs, so the fingerprint after is the one before.

What must not be handed on: a digest to a cached call made by a function
the statement called, whose body may have changed the argument first; and
the fingerprint after a call whose body ran.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash import content_hashers, handed_digests
from tests._cell_driver import run_cash_cell

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")

LIB = (
    "import cash\n"
    "import polars as pl\n"
    "@cash.cache\n"
    "def clean(df, k=1):\n"
    "    return df.with_columns(y=pl.col('x') * k)\n"
    "@cash.cache\n"
    "def total(arr):\n"
    "    return float(arr.sum())\n"
    "def bump_then_total(arr):\n"
    "    arr[0] = 1000.0\n"
    "    return total(arr)\n"
    "@cash.cache\n"
    "def widen(df):\n"
    "    df.insert_column(df.width, pl.Series('w', [0] * df.height))\n"
    "    return df.width\n"
)


@pytest.fixture
def lib(tmp_path, monkeypatch, mock_shell):
    name = f"_handed_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    mock_shell.user_ns["__name__"] = "__main__"
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def reads_of_raw(monkeypatch, mock_shell):
    """How many times the value named ``raw`` is hashed in full."""
    seen = []
    real = content_hashers.hash_polars

    def counting(value, *a, **k):
        if value is mock_shell.user_ns.get("raw"):
            seen.append(1)
        return real(value, *a, **k)

    monkeypatch.setattr(content_hashers, "hash_polars", counting)
    return seen


def _frame(cash_magics, lib):
    run_cash_cell(cash_magics, f"import {lib}\nimport polars as pl")
    run_cash_cell(cash_magics, "raw = pl.DataFrame({'x': list(range(50_000))})")


def test_a_miss_reads_the_argument_twice(cash_magics, lib, reads_of_raw):
    _frame(cash_magics, lib)
    reads_of_raw.clear()
    run_cash_cell(cash_magics, f"df = {lib}.clean(raw, 3)")
    assert len(reads_of_raw) == 2, f"raw was hashed {len(reads_of_raw)} times"
    assert cash_magics.shell.user_ns["df"]["y"][5] == 15


def test_a_hit_reads_the_argument_once(cash_magics, lib, reads_of_raw):
    _frame(cash_magics, lib)
    run_cash_cell(cash_magics, f"df = {lib}.clean(raw, 3)")
    reads_of_raw.clear()
    run_cash_cell(cash_magics, f"other = {lib}.clean(raw, 3)")
    assert len(reads_of_raw) == 1, f"raw was hashed {len(reads_of_raw)} times"
    assert cash_magics.shell.user_ns["other"]["y"][5] == 15


def test_a_call_made_by_a_function_the_statement_called_keys_its_own_read(cash_magics, lib):
    run_cash_cell(cash_magics, f"import {lib}\nimport numpy as np")
    run_cash_cell(cash_magics, "arr = np.ones(200_000)")
    run_cash_cell(cash_magics, f"before = {lib}.total(arr)")
    run_cash_cell(cash_magics, f"after = {lib}.bump_then_total(arr)")
    ns = cash_magics.shell.user_ns
    assert ns["before"] == 200_000.0
    assert ns["after"] == 200_000.0 + 999.0


def test_a_body_that_changes_its_argument_is_seen(cash_magics, lib):
    _frame(cash_magics, lib)
    lineages = cash_magics.tracking_state.variable_lineage
    before = lineages["raw"]
    run_cash_cell(cash_magics, f"n = {lib}.widen(raw)")
    assert cash_magics.shell.user_ns["raw"].width == 2
    assert lineages["raw"] != before, "the in-place change was not seen"


def test_only_the_statements_own_call_claims_the_digests():
    value = np.zeros(3)
    offered = handed_digests.offer("<cash-test>", {id(value): (value, "d")})
    try:
        assert handed_digests.claim() is None, "claimed from a frame that is not the statement's"
        code = compile("got = claim()", "<cash-test>", "exec")
        namespace = {"claim": handed_digests.claim}
        exec(code, namespace)
        assert namespace["got"] is offered
        assert offered.digest(value) == "d"
        exec(code, namespace)
        assert namespace["got"] is None, "claimed twice"
    finally:
        handed_digests.withdraw()
