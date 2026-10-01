"""Standard-library helpers that read the environment for the caller.

``os.path.expandvars("$DATA_DIR/x")``, ``os.path.expanduser("~/x")``,
``Path.home()``, ``tempfile.gettempdir()``, ``shutil.which(...)`` and
``os.environ.setdefault(...)`` read a variable, but their answer went into no
key and nothing warned: the first process's value was served to every later
one.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import warnings

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer

pytestmark = pytest.mark.core


def _quiet(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [getattr(w.message, "code", None) for w in rec]


def expandvars():
    return os.path.expandvars("$CASH_TEST_DATA_DIR/events.csv")


def expanduser():
    return os.path.expanduser("~/events.csv")


def home():
    return str(pathlib.Path.home())


def path_expanduser():
    return str(pathlib.Path("~/events.csv").expanduser())


def which():
    return shutil.which("cash-test-tool")


def tempdir():
    return tempfile.gettempdir()


def expandvars_of_a_parameter(p):
    return os.path.expandvars(p)


#: Where ``~`` comes from: Windows reads USERPROFILE and ignores HOME.
HOME_VAR = "USERPROFILE" if os.name == "nt" else "HOME"
#: Plain names, so Windows path separators cannot change how they read back.
HOMES = ("cash_home_a", "cash_home_b")


@pytest.mark.parametrize(
    ("fn", "var", "values"),
    [
        (expandvars, "CASH_TEST_DATA_DIR", ("/data/a", "/data/b")),
        (expanduser, HOME_VAR, HOMES),
        (home, HOME_VAR, HOMES),
        (path_expanduser, HOME_VAR, HOMES),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_a_new_value_is_a_new_entry(disk_cash, monkeypatch, fn, var, values):
    cached = disk_cash.cache(fn)
    monkeypatch.setenv(var, values[0])
    first, codes = _quiet(cached)
    assert codes == [] and values[0] in first
    monkeypatch.setenv(var, values[1])
    assert values[1] in cached()
    monkeypatch.setenv(var, values[0])
    assert values[0] in cached()
    assert cached.cache_info()["hits"] >= 1


def test_which_follows_path(disk_cash, monkeypatch, tmp_path):
    # Windows finds a program only by an extension listed in PATHEXT.
    suffix = ".bat" if os.name == "nt" else ""
    for sub in ("one", "two"):
        tool = tmp_path / sub / f"cash-test-tool{suffix}"
        tool.parent.mkdir()
        tool.write_text("#!/bin/sh\n", encoding="utf-8")
        tool.chmod(0o755)
    cached = disk_cash.cache(which)
    monkeypatch.setenv("PATH", str(tmp_path / "one"))
    assert "one" in cached()
    monkeypatch.setenv("PATH", str(tmp_path / "two"))
    assert "two" in cached()


def test_the_temporary_directory_is_an_input(disk_cash, monkeypatch, tmp_path):
    cached = disk_cash.cache(tempdir)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "a"))
    assert cached().endswith("a")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "b"))
    assert cached().endswith("b")


def test_setdefault_is_keyed_and_its_write_is_reported(disk_cash, monkeypatch):
    def setdefault():
        return os.environ.setdefault("CASH_TEST_MODE", "default")

    report = PurityAnalyzer().analyze(setdefault)
    assert ("env", "CASH_TEST_MODE") in report.environment_reads
    assert any("setdefault() - write method" in i.description for i in report.issues), report.issues
    cached = disk_cash.cache(setdefault, assume_safe=True)
    monkeypatch.setenv("CASH_TEST_MODE", "a")
    assert cached() == "a"
    monkeypatch.setenv("CASH_TEST_MODE", "b")
    assert cached() == "b"


def test_a_computed_argument_still_warns():
    """Control: which variables ``expandvars(p)`` reads depends on ``p``."""
    report = PurityAnalyzer().analyze(expandvars_of_a_parameter)
    assert [i.kind for i in report.issues] == ["ambient_read"], report.issues
