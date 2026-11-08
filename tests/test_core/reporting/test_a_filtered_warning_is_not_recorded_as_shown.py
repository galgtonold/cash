"""A once-per-cache warning is recorded only when it was shown.

IMPURE-SIDE-EFFECTS is shown once per cache, not once per process: the next
run on the same cache stays quiet. It was recorded before it was emitted,
whatever the warning filters did with it. So the documented CI gate,
``filterwarnings("error", category=CashImpurityWarning)``, failed the first
run and passed every rerun on the same cache, and a test suite that ignored
the warning hid it from the developer's own runs for good.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.timeout(120)

_SCRIPT = """
import sys, warnings, cash
if sys.argv[1] != "none":
    warnings.filterwarnings(sys.argv[1], category=cash.CashImpurityWarning)
else:
    warnings.simplefilter("always")

@cash.cache
def build_report(rows):
    print(f"building {len(rows)} rows")
    return sum(rows)

try:
    build_report([int(sys.argv[2])])  # a new argument: a miss every run
    print("RUN PASSED")
except cash.CashImpurityWarning:
    print("RUN FAILED")
"""


def _run(tmp_path, action, n):
    script = tmp_path / "model.py"
    script.write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    return run_python(script, action, n, cwd=tmp_path, timeout=100, check=False)


def test_an_error_filter_fails_every_run_on_the_cache(tmp_path):
    outcomes = [_run(tmp_path, "error", n).stdout.strip().splitlines()[-1] for n in (1, 2, 3)]
    assert outcomes == ["RUN FAILED"] * 3


def test_an_ignored_warning_is_shown_to_the_next_run(tmp_path):
    ignored = _run(tmp_path, "ignore", 1)
    assert "IMPURE-SIDE-EFFECTS" not in ignored.stderr
    shown = _run(tmp_path, "none", 2)
    assert "IMPURE-SIDE-EFFECTS" in shown.stderr, shown.stderr


def test_a_shown_warning_is_still_once_per_cache(tmp_path):
    """The control: shown once, the next run on the cache stays quiet."""
    assert "IMPURE-SIDE-EFFECTS" in _run(tmp_path, "none", 1).stderr
    assert "IMPURE-SIDE-EFFECTS" not in _run(tmp_path, "none", 2).stderr


def test_a_warning_the_default_filter_holds_back_is_not_reported_shown():
    """Under ``"default"`` a warning already shown at a line is held back
    the second time; reported shown, it was recorded as seen by a reader
    who never saw it."""
    import warnings

    from cash.diagnostics import warn_diagnostic_message
    from cash.exceptions import CashImpurityWarning

    def emit():
        return warn_diagnostic_message(CashImpurityWarning, "IMPURE-SIDE-EFFECTS", "a test notice")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        globals().pop("__warningregistry__", None)
        shown = [emit(), emit()]
        warnings.simplefilter("always")
        shown.append(emit())
    assert len(caught) == 2
    assert shown == [True, False, True]
