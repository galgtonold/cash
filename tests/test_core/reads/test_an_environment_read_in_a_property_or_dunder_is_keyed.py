"""An environment read inside a property or a dunder method is an input.

A settings object whose ``@property`` (or ``__getitem__``, ``__enter__``)
reads ``os.environ["MODE"]`` served the first mode's result to every later
mode, with no warning, while the same read in an ordinary method called by
name was keyed: the analysis walks only calls the code spells out. What an
object the call reads or is given runs without being named is analysed too:
its environment reads are folded into the key and a clock read warns.
"""

from __future__ import annotations

import os
import time
import warnings

import pytest

_VAR = "CASH_TEST_SETTINGS_MODE"


class Settings:
    @property
    def mode(self):
        return os.environ[_LITERAL]

    def __getitem__(self, key):
        return os.environ["CASH_TEST_SETTINGS_MODE"]

    def __enter__(self):
        return os.environ["CASH_TEST_SETTINGS_MODE"]

    def __exit__(self, *exc):
        return False

    @property
    def stamp(self):
        return time.time()


_LITERAL = "CASH_TEST_SETTINGS_MODE"
settings = Settings()


def by_property(x):
    return f"{x}-{settings.mode}"


def by_item(x):
    return f"{x}-{settings['MODE']}"


def by_with(x):
    with settings as mode:
        return f"{x}-{mode}"


def by_argument(cfg):
    return f"arg-{cfg.mode}"


@pytest.mark.parametrize(
    ("body", "arg"),
    [(by_property, 1), (by_item, 1), (by_with, 1), (by_argument, "settings")],
)
def test_a_new_value_is_a_new_entry(cash_instance, monkeypatch, body, arg):
    f = cash_instance.cache(body)
    arg = settings if arg == "settings" else arg
    monkeypatch.setenv(_VAR, "prod")
    first = f(arg)
    assert first.endswith("-prod")
    monkeypatch.setenv(_VAR, "staging")
    assert f(arg).endswith("-staging")
    monkeypatch.setenv(_VAR, "prod")
    assert f(arg) == first
    assert f.explain(arg).would_hit, "the same value did not hit"


def test_a_clock_read_in_a_property_warns(cash_instance):
    @cash_instance.cache
    def stamped(x):
        return (x, settings.stamp)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        stamped(1)
    codes = [getattr(w.message, "code", None) for w in rec]
    assert "KEY-AMBIENT-READ" in codes
