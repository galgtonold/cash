"""A one-line helper that returns the clock or an environment variable.

Its own read is left to the call site, which only judged a bare name
(``now()``): ``module.now()``, ``Class.static()`` and ``self.stamp()`` froze
the first value into every later result with no warning at all.
"""

from __future__ import annotations

import time

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer
from cash.effects import environment_component

from . import _clock_helpers_fixture as clocks

pytestmark = pytest.mark.core


def _ambient(fn):
    return [i.description for i in PurityAnalyzer().analyze(fn).issues if i.kind == "ambient_read"]


def through_the_module():
    return clocks.now()


def through_a_class():
    return clocks.Clock.static()


def env_with_a_parameter_name():
    return clocks.env("CASH_TEST_APP_MODE")


def env_through_a_named_constant():
    return clocks.mode()


_now = clocks.now


def by_value():
    fn = _now
    return fn()


class Service:
    def stamp(self):
        return time.time()

    def run(self):
        return self.stamp()


@pytest.mark.parametrize(
    ("fn", "said"),
    [
        (through_the_module, "clocks.now()"),
        (through_a_class, "clocks.Clock.static()"),
        (env_with_a_parameter_name, "clocks.env()"),
        (Service.run, "self.stamp()"),
    ],
)
def test_a_dotted_call_to_a_clock_helper_is_an_ambient_read(fn, said):
    found = _ambient(fn)
    assert any(said in d for d in found), found


def test_a_clock_helper_reached_as_a_value_reports_its_own_read():
    """Not judged at a call site (``fn = now; fn()``), so its own read stands."""

    assert any("time.time" in d for d in _ambient(by_value)), _ambient(by_value)


def test_a_bare_name_call_is_still_reported_once():
    """Control: judged at the call site, and not again inside the helper."""
    now = clocks.now

    def bare():
        return now()

    found = _ambient(bare)
    assert len(found) == 1 and "now()" in found[0], found


def test_an_environment_variable_named_by_a_module_constant_is_keyed(monkeypatch):
    report = PurityAnalyzer().analyze(env_through_a_named_constant)
    assert report.environment_reads, report.environment_reads
    assert not [i for i in report.issues if i.kind == "ambient_read"], report.issues
    monkeypatch.setenv("CASH_TEST_APP_MODE", "x")
    first = environment_component(report.environment_reads)
    monkeypatch.setenv("CASH_TEST_APP_MODE", "y")
    assert environment_component(report.environment_reads) != first
