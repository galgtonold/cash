"""Making a relative path absolute reads the working directory, and is keyed on it.

Found while stress-testing the decorator: only ``os.getcwd()`` was keyed.
``os.path.abspath("results")``, ``Path.cwd()``, ``os.path.realpath(".")`` and
``Path("x").resolve()`` read the working directory too, and the first
directory's answer was served in every other one -- a job then wrote into the
other job's folder.
"""

from __future__ import annotations

import ast
import os
import warnings
from pathlib import Path

import pytest

from cash import Cash
from cash.effects import environment_input


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def by_abspath():
    return os.path.abspath("results")


def by_path_cwd():
    return str(Path.cwd())


def by_realpath():
    return os.path.realpath(".")


def by_resolve():
    return str(Path("results").resolve())


def by_absolute(p):
    return str(Path(p).absolute())


@pytest.mark.parametrize(
    "fn", [by_abspath, by_path_cwd, by_realpath, by_resolve, by_absolute], ids=lambda f: f.__name__
)
def test_another_directory_is_another_entry(c, monkeypatch, tmp_path, fn):
    cached = c.cache(fn)
    args = ("results",) if fn is by_absolute else ()
    (tmp_path / "jobA").mkdir()
    (tmp_path / "jobB").mkdir()
    monkeypatch.chdir(tmp_path / "jobA")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "jobA" in cached(*args)
    monkeypatch.chdir(tmp_path / "jobB")
    assert "jobB" in cached(*args)


@pytest.mark.parametrize(
    "source",
    [
        "os.path.abspath(__file__)",
        "os.path.abspath('/srv/data')",
        "Path(__file__).resolve()",
        "Path(__file__).parent.resolve()",
        "(Path(__file__).parent / 'data').resolve()",
        "os.path.realpath(os.path.dirname(__file__))",
        "promise.resolve(1)",
    ],
)
def test_a_path_that_is_already_absolute_reads_no_working_directory(source):
    """``os.path.dirname(os.path.abspath(__file__))`` is how code finds its own
    data; keying it on the working directory would cost a miss per directory."""
    assert environment_input(ast.parse(source, mode="eval").body) is None
