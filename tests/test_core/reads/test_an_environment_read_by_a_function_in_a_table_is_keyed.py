"""A function reached through a table or a step list keys what it reads from
the environment, and a clock read in it warns.

``HANDLERS = {"tag": tag}`` with ``tag`` reading ``os.environ["MODE"]``, or
``STEPS = [lambda x: x + os.environ["MODE"]]``: the module constants such a
function reads were keyed, its environment reads were not, so the next run
with another value got the old result; a clock read in it was frozen with
no warning. The same helper called by name was keyed all along.
"""

from __future__ import annotations

import os
import time
import warnings

import pytest

_VAR = "CASH_TEST_TABLE_MODE"


def tag(x):
    return f"{x}-" + os.environ["CASH_TEST_TABLE_MODE"]


def stamp(x):
    return f"{x}@{time.time()}"


HANDLERS = {"tag": tag}
STEPS = [lambda x: f"{x}-" + os.environ["CASH_TEST_TABLE_MODE"]]
CLOCKED = {"stamp": stamp}


def dispatch(x):
    return HANDLERS["tag"](x)


def pipeline(x):
    for step in STEPS:
        x = step(x)
    return x


def passed(fn, x):
    return fn(x)


@pytest.mark.parametrize("body", [dispatch, pipeline])
def test_a_new_value_is_a_new_entry(cash_instance, monkeypatch, body):
    f = cash_instance.cache(body)
    monkeypatch.setenv(_VAR, "prod")
    assert f(1) == "1-prod"
    monkeypatch.setenv(_VAR, "staging")
    assert f(1) == "1-staging"
    monkeypatch.setenv(_VAR, "prod")
    assert f(1) == "1-prod"
    assert f.explain(1).would_hit, "the same value did not hit"


def test_a_function_passed_as_an_argument_keys_it_too(cash_instance, monkeypatch):
    f = cash_instance.cache(passed)
    monkeypatch.setenv(_VAR, "prod")
    assert f(tag, 1) == "1-prod"
    monkeypatch.setenv(_VAR, "staging")
    assert f(tag, 1) == "1-staging"


def test_a_clock_read_through_a_table_warns(cash_instance):
    @cash_instance.cache
    def clocked(x):
        return CLOCKED["stamp"](x)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        clocked(1)
    assert "KEY-AMBIENT-READ" in [getattr(w.message, "code", None) for w in rec]
