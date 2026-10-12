"""A statement that makes a path from the working directory keys on it.

``pd.read_csv(Path.cwd() / 'data.csv')``, ``os.path.abspath('data.csv')``
and ``Path('data.csv').resolve()`` read the working directory, which the
key folds (``("cwd", "")``) -- but the notebook only looked for it in a
statement spelling ``environ``, ``getenv`` or ``getcwd``. After the
``os.chdir`` cell was edited, the old folder's data was served.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from cash.notebook.lineage_formula import statement_environment_reads

NS = {"os": os, "pathlib": pathlib, "Path": pathlib.Path}


@pytest.mark.parametrize(
    "code",
    [
        "k = read(pathlib.Path.cwd() / 'data.csv')",
        "k = read(Path.cwd() / 'data.csv')",
        "b = read(os.path.abspath('data.csv'))",
        "b = read(os.path.realpath('data.csv'))",
        "n = read(pathlib.Path('data.csv').resolve())",
        "n = read(Path('data.csv').absolute())",
    ],
)
def test_the_working_directory_is_read(code):
    assert ("cwd", "") in statement_environment_reads(code, NS)


@pytest.mark.parametrize(
    ("code", "entry"),
    [
        ("h = Path.home() / 'x'", ("expanduser", "~")),
        ("p = os.path.expanduser('~/x')", ("expanduser", "~")),
    ],
)
def test_the_other_helpers_are_read(code, entry):
    assert entry in statement_environment_reads(code, NS)


def test_an_absolute_path_reads_no_working_directory():
    assert statement_environment_reads("n = read(Path('/data/x.csv').resolve())", NS) == set()
