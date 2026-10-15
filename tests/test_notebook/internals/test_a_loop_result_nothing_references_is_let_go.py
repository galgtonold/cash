"""A loop's call result nothing references any more is let go at once.

``h, _ = fit(df, k)`` in a loop: the call cache held every call's result
until the next cell (so a statement of the cell holding one stores a
reference to its entry), and the tuple each pass unpacks kept that pass's
frame alive. A 15-pass grid held 430 MB, every frame one more sharer of
``df``'s columns for the frame memo to walk. A held result nothing else
references can reach no later statement, so it now goes at the next
statement; one a variable still holds stays.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

MODULE = f"""
import time


def fit(frame, k):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return frame.assign(score=frame["x"] * k), {{"k": k}}
"""


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_fit_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module(name).__name__
    sys.modules.pop(name, None)


def _held(processor) -> list:
    """The call results the call cache holds."""
    return [entry[0] for entry in processor._calls.held_call_results()]


def test_each_pass_result_goes_once_unbound(cash_magics, mock_shell, lib, statement_processor):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(
        cash_magics,
        f"import {lib} as m\nimport numpy as np, pandas as pd\ndf = pd.DataFrame({{'x': np.arange(50_000.0)}})",
    )
    run_cash_cell(
        cash_magics,
        "kept = []\n"
        "totals = []\n"
        "for k in range(6):\n"
        "    h, _ = m.fit(df, k)\n"
        "    totals.append(float(h['score'].sum()))\n"
        "    if k == 2:\n"
        "        kept.append(h)\n"
        "whole = m.fit(df, 9)\n"
        "done = len(totals)\n",
    )
    ns = mock_shell.user_ns
    assert ns["totals"] == [float(np.arange(50_000.0).sum() * k) for k in range(6)]
    held = _held(statement_processor)
    # The passes' tuples were unpacked and dropped; ``whole`` still holds its
    # own (the positive control: results are held at all).
    assert [id(value) for value in held] == [id(ns["whole"])]
