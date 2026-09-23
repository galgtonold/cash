"""An environment read is an input: its value is folded into the key.

``os.environ["TENANT"]`` in a cached body used to serve the first tenant's
answer to every other tenant, with a KEY-AMBIENT-READ warning in the decorated
function and nothing at all in a notebook statement. Both paths now read the
variable's current value on every call and fold it into the key, so a new
value is a new entry and there is nothing to warn about. A read whose name is
only known at run time cannot be folded, and still warns in a decorated
function.
"""

from __future__ import annotations

import logging
import os
import warnings

import pytest

from cash import Cash
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.notebook.cache_status import CacheStatus
from cash.notebook.lineage_formula import statement_environment_reads
from cash.purity_analyzer import ISSUE_AMBIENT_READ, PurityAnalyzer
from tests._cell_driver import run_cash_cell

_VAR = "CASH_TEST_TENANT"


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _codes(fn, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*args)
    return [getattr(w.message, "code", None) for w in rec]


def by_subscript():
    return f"report for {os.environ[_TENANT]}"


def by_getenv():
    return f"report for {os.getenv('CASH_TEST_TENANT')}"


def by_environ_get():
    return f"report for {os.environ.get('CASH_TEST_TENANT', 'nobody')}"


def by_literal_subscript():
    return f"report for {os.environ['CASH_TEST_TENANT']}"


def in_the_working_directory():
    return os.path.basename(os.getcwd())


def by_runtime_name(name):
    return os.getenv(name)


def _tenant():
    return os.environ.get("CASH_TEST_TENANT", "nobody")


def through_a_helper():
    return _tenant().upper()


_TENANT = _VAR


@pytest.mark.parametrize("fn", [by_getenv, by_environ_get, by_literal_subscript], ids=lambda f: f.__name__)
def test_a_new_value_is_a_new_entry(c, monkeypatch, fn):
    cached = c.cache(fn)
    monkeypatch.setenv(_VAR, "acme")
    assert _codes(cached) == []
    assert cached() == fn()
    monkeypatch.setenv(_VAR, "globex")
    assert cached() == fn() and "globex" in cached()
    monkeypatch.setenv(_VAR, "acme")
    assert "acme" in cached()
    assert cached.cache_info()["hits"] >= 1


def test_a_read_in_a_helper_is_keyed(c, monkeypatch):
    cached = c.cache(through_a_helper)
    monkeypatch.setenv(_VAR, "acme")
    assert _codes(cached) == []
    monkeypatch.setenv(_VAR, "globex")
    assert cached() == "GLOBEX"


def test_the_working_directory_is_an_input(c, monkeypatch, tmp_path):
    cached = c.cache(in_the_working_directory)
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    monkeypatch.chdir(tmp_path / "one")
    assert _codes(cached) == []
    assert cached() == "one"
    monkeypatch.chdir(tmp_path / "two")
    assert cached() == "two"


def test_a_name_only_known_at_run_time_still_warns(c, monkeypatch):
    monkeypatch.setenv(_VAR, "acme")
    assert "KEY-AMBIENT-READ" in _codes(c.cache(by_runtime_name), _VAR)
    assert [i.kind for i in PurityAnalyzer().analyze(by_runtime_name).issues] == [ISSUE_AMBIENT_READ]


def test_a_subscript_through_a_constant_name_still_warns():
    """`os.environ[_TENANT]`: the name is a global, not written out."""
    assert [i.kind for i in PurityAnalyzer().analyze(by_subscript).issues] == [ISSUE_AMBIENT_READ]


def test_the_analyzer_lists_what_the_key_folds():
    assert PurityAnalyzer().analyze(by_getenv).environment_reads == {("env", _VAR)}
    assert PurityAnalyzer().analyze(in_the_working_directory).environment_reads == {("cwd", "")}
    assert PurityAnalyzer().analyze(by_getenv).issues == ()


def test_a_read_in_a_cached_dependency_reaches_the_caller(c, monkeypatch):
    """The dependency's own key moves with the variable; the caller's stored
    result must not outlive it."""
    tenant = c.cache(by_getenv)

    def headline():
        return tenant().upper()

    cached = c.cache(headline, depends_on=[tenant])
    monkeypatch.setenv(_VAR, "acme")
    assert cached() == "REPORT FOR ACME"
    monkeypatch.setenv(_VAR, "globex")
    assert cached() == "REPORT FOR GLOBEX"


# -- the notebook key ------------------------------------------------------------


def _key(code, user_ns=None):
    return compute_cache_key(
        code, set(), ctx=CacheKeyContext(variable_lineage={}, user_ns=user_ns or {"os": os})
    ).cache_key


@pytest.mark.parametrize(
    "code",
    [
        "t = os.getenv('CASH_TEST_TENANT')",
        "t = os.environ['CASH_TEST_TENANT']",
        "t = os.environ.get('CASH_TEST_TENANT')",
    ],
)
def test_a_notebook_key_folds_the_value(monkeypatch, code):
    monkeypatch.setenv(_VAR, "acme")
    first = _key(code)
    assert _key(code) == first
    monkeypatch.setenv(_VAR, "globex")
    assert _key(code) != first
    monkeypatch.delenv(_VAR)
    assert _key(code) != first


def test_a_notebook_key_without_an_environment_read_is_unchanged(monkeypatch):
    code = "t = len('CASH_TEST_TENANT')"
    first = _key(code)
    monkeypatch.setenv(_VAR, "globex")
    assert _key(code) == first
    assert statement_environment_reads(code) == set()


def test_the_notebook_resolves_an_alias():
    import os as _os

    assert statement_environment_reads("t = o.getenv('X')", {"o": _os}) == {("env", "X")}
    assert statement_environment_reads("t = os.getcwd()", {"os": os}) == {("cwd", "")}
    assert statement_environment_reads("t = os.getenv(name)", {"os": os}) == set()


@pytest.fixture
def magics(cash_magics, mock_shell):
    # The notebook's module is ``__main__``, as in a kernel.
    mock_shell.user_ns["__name__"] = "__main__"
    return cash_magics


def _restored(magics) -> bool:
    """Whether the last cell restored any statement, as ``%cash_status`` reports it."""
    rows = magics.cash_status("dict")["last_cell"]["statements"]
    return any(row["status"] == CacheStatus.RESTORED for row in rows)


def test_what_is_built_on_the_read_follows_the_value(magics, monkeypatch):
    """The key alone was not enough: the read re-ran for a new tenant, but its
    output kept its lineage, so the statement below hit and returned the
    first tenant's answer."""
    m = magics
    read = "import os, time\ntenant = (time.sleep(0.02), os.getenv('CASH_TEST_TENANT'))[1]"
    shout = "import time\nloud = (time.sleep(0.02), tenant.upper())[1]"
    for run, value in enumerate(("acme", "globex", "acme")):
        monkeypatch.setenv(_VAR, value)
        run_cash_cell(m, read)
        read_restored = _restored(m)
        run_cash_cell(m, shout)
        assert m.shell.user_ns["loud"] == value.upper()
        restored = read_restored, _restored(m)
        assert restored == ((True, True) if run == 2 else (False, False)), restored


def test_a_miss_names_the_variable(tmp_path, monkeypatch, caplog):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, verbose=True)
    cached = c.cache(by_getenv)
    monkeypatch.setenv(_VAR, "acme")
    cached()
    monkeypatch.setenv(_VAR, "globex")
    with caplog.at_level(logging.DEBUG, logger="cash.calls"):
        cached()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert f"environment variable {_VAR} changed" in text, text
    assert "globex" not in text, "the value itself reached the log"
