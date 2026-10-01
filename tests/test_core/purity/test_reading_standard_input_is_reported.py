"""Reading standard input is reported, like ``input()``.

``sys.stdin.read()`` in a cached function froze the first run's input into
every later run, with no warning: only ``input()`` was on the list.
"""

from __future__ import annotations

import json
import sys

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer

pytestmark = pytest.mark.core


def _said(fn):
    return [i.description for i in PurityAnalyzer().analyze(fn).issues]


def read_all():
    return sys.stdin.read()


def read_line():
    return sys.stdin.readline()


def read_bytes():
    return sys.stdin.buffer.read()


def iterate():
    return [line for line in sys.stdin]


def load_json():
    return json.load(sys.stdin)


def is_interactive():
    return sys.stdin.isatty()


@pytest.mark.parametrize("fn", [read_all, read_line, read_bytes, iterate, load_json], ids=lambda f: f.__name__)
def test_a_read_of_standard_input_is_reported(fn):
    said = _said(fn)
    assert any("sys.stdin" in d for d in said), said


def test_asking_whether_stdin_is_a_terminal_is_not():
    """Control: `.isatty()` reads nothing."""
    assert _said(is_interactive) == []
