"""``# @cash:assume-safe`` caches a statement whose helper cash cannot analyse.

When the analysis of a function the statement calls raises, cash does not
know what that function reads from the environment, the clock or a fresh
id, and the statement runs every time (``NOTEBOOK-ANALYSIS-FAILED``). The
function's CODE is still in the key by another path: the functions a
statement reaches are found from their bytecode, one defined in a cell is
keyed by the lineage of its name, one of a local module by the module's
source. So a waiver, which accepts the reads, lets the statement be cached,
and an edit to the helper or to what it calls still runs it again.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from cash.analysis import cacheability_decision
from cash.analysis.annotations import CacheAnnotation
from cash.notebook import lineage_formula
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

_PERSIST = CacheAnnotation(persist=True)
_WAIVED = CacheAnnotation(persist=True, assume_safe=True)

INNER = "def inner(v):\n    return v + 1\n"
HELPER = "import os\n\ndef helper(v):\n    return inner(v) * 10 + len(os.environ.get('CASH_T_MODE', ''))\n"

STEPS = "def inner(v):\n    return v + 1\n"
MODULE = (
    "import os\nfrom {steps} import inner\n\n"
    "def twice(v):\n    return v * 2\n\n"
    "def helper(v):\n    return twice(inner(v)) * 10 + len(os.environ.get('CASH_T_MODE', ''))\n"
)
#: The same ``helper``, built by the module from text its file does not hold.
BUILT = "import os\nexec(open(__file__ + '.txt').read())\n"
BUILT_TEXT = "def helper(v):\n    return v * 10 + len(os.environ.get('CASH_T_MODE', ''))\n"


@pytest.fixture(autouse=True)
def _report_every_failure(monkeypatch):
    monkeypatch.setattr(cacheability_decision, "_REPORTED_FAILURES", set())
    monkeypatch.setenv("CASH_T_MODE", "a")


@pytest.fixture
def unreadable(monkeypatch):
    """The analysis of ``helper`` raises."""
    real = lineage_formula.get_analyzer()

    class _Analyzer:
        def analyze(self, fn):
            if getattr(fn, "__name__", "") == "helper":
                raise RuntimeError("a construct the analysis cannot read")
            return real.analyze(fn)

    monkeypatch.setattr(lineage_formula, "get_analyzer", _Analyzer)


def _cell(magics, code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(magics, code)


def _run(processor, code, annotation=_WAIVED):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = processor.process_statement(code, annotation=annotation)
    return result, [w.message for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-ANALYSIS-FAILED"]


class TestAFunctionOfACell:
    @pytest.fixture
    def nb(self, cash_magics, mock_shell, unreadable):
        _cell(cash_magics, INNER)
        _cell(cash_magics, HELPER)
        return mock_shell.user_ns

    def test_waived_it_is_cached_without_a_warning(self, nb, statement_processor):
        first, said = _run(statement_processor, "r = helper(1)")
        again, said_again = _run(statement_processor, "r = helper(1)")
        assert first["status"] == CacheStatus.COMPUTED
        assert again["status"] != CacheStatus.COMPUTED, again.get("uncacheable_reasons")
        assert nb["r"] == 21
        assert said == said_again == []

    def test_an_edit_to_the_helper_runs_it_again(self, nb, cash_magics, statement_processor):
        _run(statement_processor, "r = helper(1)")
        _cell(cash_magics, HELPER.replace("* 10", "* 100"))
        after, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 201, "served for the helper as it was"
        assert after["status"] == CacheStatus.COMPUTED

    def test_an_edit_to_a_function_it_calls_runs_it_again(self, nb, cash_magics, statement_processor):
        _run(statement_processor, "r = helper(1)")
        _cell(cash_magics, INNER.replace("v + 1", "v + 2"))
        after, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 31, "served for the function it calls as it was"
        assert after["status"] == CacheStatus.COMPUTED

    def test_a_changed_environment_variable_runs_it_again(self, nb, statement_processor, monkeypatch):
        _run(statement_processor, "r = helper(1)")
        monkeypatch.setenv("CASH_T_MODE", "bbb")
        after, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 23, "served for the variable as it was"
        assert after["status"] == CacheStatus.COMPUTED

    def test_without_a_waiver_it_runs_every_time_and_the_fix_names_the_waiver(self, nb, statement_processor):
        first, said = _run(statement_processor, "r = helper(1)", _PERSIST)
        again, _ = _run(statement_processor, "r = helper(1)", _PERSIST)
        assert first["status"] == again["status"] == CacheStatus.COMPUTED
        assert len(said) == 1
        assert "@cash:assume-safe" in str(said[0])
        assert "still in the key" in str(said[0])


def test_a_waiver_on_the_def_line_caches_its_callers(cash_magics, mock_shell, unreadable, statement_processor):
    _cell(cash_magics, INNER)
    _cell(cash_magics, HELPER.replace("def helper(v):", "def helper(v):  # @cash:assume-safe"))
    first, said = _run(statement_processor, "r = helper(1)", _PERSIST)
    again, _ = _run(statement_processor, "r = helper(1)", _PERSIST)
    assert first["status"] == CacheStatus.COMPUTED
    assert again["status"] != CacheStatus.COMPUTED, again.get("uncacheable_reasons")
    assert said == []


def test_every_unread_function_must_be_waived(cash_magics, mock_shell, monkeypatch, statement_processor):
    """One waived on its ``def`` line does not speak for another the
    statement calls."""
    real = lineage_formula.get_analyzer()

    class _Analyzer:
        def analyze(self, fn):
            if getattr(fn, "__name__", "") in ("helper", "other"):
                raise RuntimeError("a construct the analysis cannot read")
            return real.analyze(fn)

    monkeypatch.setattr(lineage_formula, "get_analyzer", _Analyzer)
    _cell(cash_magics, INNER)
    _cell(cash_magics, HELPER.replace("def helper(v):", "def helper(v):  # @cash:assume-safe"))
    _cell(cash_magics, "def other(v):\n    return v - 1\n")
    _run(statement_processor, "r = helper(1) + other(1)", _PERSIST)
    again, _ = _run(statement_processor, "r = helper(1) + other(1)", _PERSIST)
    assert again["status"] == CacheStatus.COMPUTED
    assert any("other()" in reason for reason in again["uncacheable_reasons"])
    _run(statement_processor, "r = helper(1) + other(1)", _WAIVED)
    waived, _ = _run(statement_processor, "r = helper(1) + other(1)", _WAIVED)
    assert waived["status"] != CacheStatus.COMPUTED


class TestAFunctionOfALocalModule:
    @pytest.fixture
    def nb(self, cash_magics, mock_shell, tmp_path, monkeypatch, unreadable):
        tag = tmp_path.name.replace("-", "_")
        self.steps = tmp_path / f"steps_{tag}.py"
        self.steps.write_text(STEPS, encoding="utf-8")
        self.path = tmp_path / f"work_{tag}.py"
        self.path.write_text(MODULE.format(steps=f"steps_{tag}"), encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        for name in (f"steps_{tag}", f"work_{tag}"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        self.cash_magics = cash_magics
        self.importing = f"from work_{tag} import helper"
        _cell(cash_magics, self.importing)
        return mock_shell.user_ns

    def _edit(self, path, old, new):
        text = path.read_text("utf-8")
        assert old in text
        path.write_text(text.replace(old, new), encoding="utf-8")
        _cell(self.cash_magics, self.importing)

    def test_waived_it_is_cached(self, nb, statement_processor):
        _run(statement_processor, "r = helper(1)")
        again, said = _run(statement_processor, "r = helper(1)")
        assert again["status"] != CacheStatus.COMPUTED, again.get("uncacheable_reasons")
        assert nb["r"] == 41
        assert said == []

    def test_an_edit_to_the_helper_runs_it_again(self, nb, statement_processor):
        _run(statement_processor, "r = helper(1)")
        self._edit(self.path, "* 10 +", "* 100 +")
        after, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 401
        assert after["status"] == CacheStatus.COMPUTED

    def test_an_edit_to_a_function_it_calls_runs_it_again(self, nb, statement_processor):
        _run(statement_processor, "r = helper(1)")
        self._edit(self.path, "return v * 2", "return v * 3")
        after, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 61
        assert after["status"] == CacheStatus.COMPUTED

    def test_an_edit_to_a_function_of_another_module_it_calls_runs_it_again(self, nb, statement_processor):
        _run(statement_processor, "r = helper(1)")
        self._edit(self.steps, "return v + 1", "return v + 2")
        after, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 61
        assert after["status"] == CacheStatus.COMPUTED

    def test_a_changed_environment_variable_runs_it_again(self, nb, statement_processor, monkeypatch):
        _run(statement_processor, "r = helper(1)")
        monkeypatch.setenv("CASH_T_MODE", "bbb")
        after, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 43
        assert after["status"] == CacheStatus.COMPUTED


class TestCodeCashCannotSee:
    """The waiver accepts what a function reads. It cannot stand in for the
    function's code: where no key holds that, the statement stays uncached."""

    @pytest.fixture
    def nb(self, cash_magics, mock_shell, tmp_path, monkeypatch, unreadable):
        name = f"built_{tmp_path.name.replace('-', '_')}"
        path = tmp_path / f"{name}.py"
        path.write_text(BUILT, encoding="utf-8")
        (tmp_path / f"{name}.py.txt").write_text(BUILT_TEXT, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, name, raising=False)
        _cell(cash_magics, f"from {name} import helper")
        return mock_shell.user_ns

    def test_a_waiver_does_not_cache_it_and_the_warning_says_why(self, nb, statement_processor):
        first, said = _run(statement_processor, "r = helper(1)")
        again, _ = _run(statement_processor, "r = helper(1)")
        assert nb["r"] == 11
        assert first["status"] == again["status"] == CacheStatus.COMPUTED
        assert len(said) == 1
        text = str(said[0])
        assert "not built from its module's source file" in text
        assert "does not cache this statement" in text
        assert any("cannot tell when its code changes" in reason for reason in again["uncacheable_reasons"])

    def test_which_code_no_key_holds(self, nb, tmp_path):
        assert lineage_formula.unkeyed_code(nb["helper"], nb) == "it was not built from its module's source file"
        exec("def made_here(v):\n    return v\n", nb)  # as a cell would
        assert lineage_formula.unkeyed_code(nb["made_here"], nb) is None
        assert lineage_formula.unkeyed_code(len, nb) == "it is not a Python function"
