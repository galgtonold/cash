"""A statement calling a helper built by ``exec`` that reads the environment runs again.

``exec("def helper(x):\\n    return x + os.environ['MODE']", ns)``: the
helper has no source, and the analysis that keys a variable a helper reads
found none. The check for functions with no text asks their bytecode, and a
helper that makes no call reads like one that changes nothing: the
statement was stored, and after ``MODE`` changed it gave the old value, with
or without ``# @cash:assume-safe``. Restart and Run All in plain Python
gives the new one. The bytecode shows it may read what the key misses (an
attribute here), so the statement runs every time, and says so once, as
for one built by a local module. A helper built so that only computes on
its argument is still cached.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

_PERSIST = CacheAnnotation(persist=True)
_WAIVED = CacheAnnotation(persist=True, assume_safe=True)

BUILT = "import os\nns = {{'os': os}}\nexec({text!r}, ns)\nhelper = ns['helper']\n"


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    monkeypatch.setenv("CASH_T_EXEC_MODE", "a")


def _cell(magics, code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(magics, code)


def _run(processor, code, annotation):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = processor.process_statement(code, annotation=annotation)
    return result, [getattr(w.message, "code", None) for w in caught]


@pytest.mark.parametrize("annotation", [_PERSIST, _WAIVED], ids=["plain", "waived"])
@pytest.mark.parametrize(
    "text",
    [
        "def helper(x):\n    return x + os.environ['CASH_T_EXEC_MODE']",
        "helper = lambda x: x + os.environ['CASH_T_EXEC_MODE']",
    ],
    ids=["def", "lambda"],
)
def test_a_changed_variable_is_not_served_the_old_value(
    cash_magics, mock_shell, statement_processor, monkeypatch, text, annotation
):
    _cell(cash_magics, BUILT.format(text=text))
    _, said = _run(statement_processor, "r = helper('1')", annotation)
    assert said == ["NOTEBOOK-UNSEEN-WRITES"]
    monkeypatch.setenv("CASH_T_EXEC_MODE", "b")
    again, _ = _run(statement_processor, "r = helper('1')", annotation)
    assert mock_shell.user_ns["r"] == "1b", "served the value of the old variable"
    assert again["status"] == CacheStatus.COMPUTED
    assert any("what it reads or changes" in reason for reason in again["uncacheable_reasons"])


def test_one_built_by_a_local_module_runs_every_time(
    cash_magics, mock_shell, statement_processor, monkeypatch, tmp_path
):
    name = f"built_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(
        "import os\nexec(\"def helper(x):\\n    return x + os.environ['CASH_T_EXEC_MODE']\", globals())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    _cell(cash_magics, f"from {name} import helper\n")
    try:
        _, said = _run(statement_processor, "r = helper('1')", _PERSIST)
        monkeypatch.setenv("CASH_T_EXEC_MODE", "b")
        again, _ = _run(statement_processor, "r = helper('1')", _PERSIST)
    finally:
        sys.modules.pop(name, None)
    assert said == ["NOTEBOOK-UNSEEN-WRITES"], "said once, by one check"
    assert mock_shell.user_ns["r"] == "1b"
    assert again["status"] == CacheStatus.COMPUTED


def test_one_that_only_computes_on_its_argument_is_cached(cash_magics, statement_processor):
    _cell(cash_magics, BUILT.format(text="def helper(x):\n    return x * 2 + len(str(x))"))
    _, said = _run(statement_processor, "r = helper(3)", _PERSIST)
    again, _ = _run(statement_processor, "r = helper(3)", _PERSIST)
    assert said == []
    assert again["status"] != CacheStatus.COMPUTED, again.get("uncacheable_reasons")
