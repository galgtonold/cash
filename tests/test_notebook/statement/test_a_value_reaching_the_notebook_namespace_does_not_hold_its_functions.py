"""A value that reaches the notebook's namespace through a library object
does not hold every notebook function.

``pbar = pyprind.ProgBar(n, stream=sys.stdout)`` holds the kernel's output
stream, whose session leads to the kernel, its shell, and the shell's
``user_ns``. The walk for the functions a value holds went on into that
namespace and found every function of the notebook: one using ``globals()``,
called or not, made every statement reading ``pbar`` run every time (E10's
loop loading 50,000 reviews, about 230 s), and the globals of all of them
entered the key. The walk stops at the standard streams and the shell; a
namespace a value holds of its own is still walked.
"""

from __future__ import annotations

import sys

import pytest

from cash.notebook.cache_key import unkeyed_reach
from cash.notebook.held_functions import held_notebook_code


class _Shell:
    def __init__(self, user_ns):
        self.user_ns = user_ns


class _Session:
    def __init__(self, shell):
        self.parent = shell


class _Stream:
    def __init__(self, session):
        self.session = session


class _ProgBar:
    def __init__(self, stream):
        self.stream = stream
        self.max_iter = 10


@pytest.fixture
def ns():
    ns: dict = {}
    exec("def by_name(k):\n    return globals()[k]\ndef helper(x):\n    return x + 1\n", ns)
    # Held as ``sys.stdout`` is: the stream the program writes to (set
    # inside each test, as pytest's capture swaps it in after fixtures).
    ns["pbar"] = _ProgBar(_Stream(_Session(_Shell(ns))))
    return ns


@pytest.fixture
def stdout_is_the_stream(ns, monkeypatch):
    def install():
        monkeypatch.setattr(sys, "stdout", ns["pbar"].stream)

    return install


def test_a_progress_bar_holding_the_output_stream_holds_no_notebook_function(ns, stdout_is_the_stream):
    stdout_is_the_stream()
    assert held_notebook_code(ns["pbar"], ns).codes == ()


def test_a_statement_reading_it_is_not_refused_for_a_lookup_by_string(ns, stdout_is_the_stream):
    stdout_is_the_stream()
    reach = unkeyed_reach("pbar.update()", ["pbar"], ns, {"pbar": "L1"})
    assert not reach.by_string and not reach.unfollowed


def test_a_function_the_value_holds_itself_is_still_found(ns, stdout_is_the_stream):
    stdout_is_the_stream()
    ns["pbar"].callback = ns["by_name"]
    assert held_notebook_code(ns["pbar"], ns).codes == (ns["by_name"].__code__,)
    assert "pbar" in unkeyed_reach("pbar.update()", ["pbar"], ns, {"pbar": "L1"}).by_string


def test_a_namespace_a_value_holds_of_its_own_is_still_walked(ns, stdout_is_the_stream):
    stdout_is_the_stream()
    box = _Shell(ns)
    assert ns["by_name"].__code__ in held_notebook_code(box, ns).codes
