"""The command line is an input: ``sys.argv`` is folded into the key.

``python app.py 4`` after ``python app.py 0`` returned the first run's
result when a cached function, or a helper it calls, read ``sys.argv`` or an
``argparse`` parser's ``parse_args()``: the command line reached no key and
nothing warned. It is folded like an environment variable now.
"""

from __future__ import annotations

import argparse
import sys

import pytest


def _threshold():
    return float(sys.argv[1])


def _direct(xs):
    return [x for x in xs if x > float(sys.argv[1])]


def _via_helper(xs):
    return [x for x in xs if x > _threshold()]


def _via_argparse(xs):
    parser = argparse.ArgumentParser()
    parser.add_argument("t", type=float)
    return [x for x in xs if x > parser.parse_args().t]


@pytest.fixture
def command_line():
    """Sets the command line in place, so ``from sys import argv`` sees it."""
    saved = list(sys.argv)

    def set_to(*args):
        sys.argv[:] = ["app.py", *args]

    yield set_to
    sys.argv[:] = saved


@pytest.mark.parametrize("body", [_direct, _via_helper, _via_argparse])
def test_another_command_line_is_another_entry(cash_instance, command_line, body):
    f = cash_instance.cache(body)
    command_line("0")
    assert f((1, 5, 10)) == [1, 5, 10]
    command_line("4")
    assert f((1, 5, 10)) == [5, 10]
    command_line("0")
    assert f((1, 5, 10)) == [1, 5, 10]
    assert f.explain((1, 5, 10)).would_hit, "the same command line did not hit"


def test_a_parser_given_its_own_list_reads_no_command_line(cash_instance, command_line):
    """``parse_args([...])`` reads only what it is given: one entry for
    every command line, while ``parse_args()`` beside it keys apart."""
    runs = []
    reads_it = cash_instance.cache(_via_argparse)

    @cash_instance.cache
    def f(xs):
        runs.append(1)
        parser = argparse.ArgumentParser()
        parser.add_argument("t", type=float)
        return [x for x in xs if x > parser.parse_args(["4"]).t]

    command_line("0")
    f((1, 5, 10))
    reads_it((1, 5, 10))
    command_line("9")
    f((1, 5, 10))
    assert len(runs) == 1
    assert reads_it((1, 5, 10)) == [10]
