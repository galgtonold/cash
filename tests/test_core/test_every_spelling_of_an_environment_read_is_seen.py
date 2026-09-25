"""Every spelling of an environment read is keyed, or reported when it cannot be.

Found while stress-testing the decorator: only ``os.getenv("X")``,
``os.environ.get("X")`` and ``os.environ["X"]`` were recognised.
``"APP_DEBUG" in os.environ``, ``os.environ.copy()``, ``.items()``,
``dict(os.environ)`` and ``os.environb`` were neither folded into the key nor
reported, so the first run's environment was served to every later one in
silence.
"""

from __future__ import annotations

import os
import warnings

import pytest

from cash import Cash
from cash.analysis.purity_analyzer import ISSUE_AMBIENT_READ, PurityAnalyzer
from cash.notebook.lineage_formula import statement_environment_reads

_VAR = "CASH_TEST_FLAG"


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _codes(fn, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*args)
    return [getattr(w.message, "code", None) for w in rec]


def by_membership():
    return "debug" if "CASH_TEST_FLAG" in os.environ else "normal"


def by_non_membership():
    return "normal" if "CASH_TEST_FLAG" not in os.environ else "debug"


def by_bytes_subscript():
    return os.environb.get(b"CASH_TEST_FLAG", b"").decode()


def by_copy():
    return os.environ.copy().get("CASH_TEST_FLAG")


def by_items():
    return {k: v for k, v in os.environ.items() if k == "CASH_TEST_FLAG"}


def by_dict():
    return dict(os.environ).get("CASH_TEST_FLAG")


def by_membership_of_a_computed_name(name):
    return name in os.environ


def writes_and_reads_by_name():
    os.environ.setdefault("CASH_TEST_OTHER", "1")
    return os.environ.get("CASH_TEST_FLAG")


@pytest.mark.parametrize("fn", [by_membership, by_non_membership, by_bytes_subscript], ids=lambda f: f.__name__)
def test_a_read_by_name_is_keyed(c, monkeypatch, fn):
    cached = c.cache(fn)
    monkeypatch.delenv(_VAR, raising=False)
    assert _codes(cached) == []
    first = cached()
    monkeypatch.setenv(_VAR, "1")
    assert cached() == fn() != first


@pytest.mark.parametrize("fn", [by_copy, by_items, by_dict, by_membership_of_a_computed_name], ids=lambda f: f.__name__)
def test_a_read_of_the_whole_environment_is_reported(c, monkeypatch, fn):
    monkeypatch.setenv(_VAR, "1")
    args = (_VAR,) if fn is by_membership_of_a_computed_name else ()
    assert "KEY-AMBIENT-READ" in _codes(c.cache(fn), *args)
    assert ISSUE_AMBIENT_READ in [i.kind for i in PurityAnalyzer().analyze(fn).issues]


def test_changing_the_environment_is_not_a_read_of_it():
    """``setdefault`` is a side effect, reported as one; it reads no value
    back into the result."""
    report = PurityAnalyzer().analyze(writes_and_reads_by_name)
    assert ISSUE_AMBIENT_READ not in [i.kind for i in report.issues]
    assert ("env", _VAR) in report.environment_reads


def test_a_notebook_statement_folds_a_membership_test():
    assert statement_environment_reads('x = "CASH_TEST_FLAG" in os.environ', {"os": os}) == {("env", _VAR)}
