"""Impurity advisories that were wrong on ordinary code, and the ones that are not.

CAS-122, round 17, four of five testers. The warnings fired on local, harmless
code on every run -- and in one report an unrelated false positive
(`perf_counter` in a log line) buried the network warning the tester needed.
Each silenced pattern below has a control next to it that must still warn: the
point is to stop the noise, not to go quiet on the code the warning is for.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash
from cash.purity_analyzer import PurityAnalyzer

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core

GLOBAL_ROWS: list = []


def read_export(path):
    return {"qty": [1, 2]}


def _issues(fn):
    return list(PurityAnalyzer().analyze(fn).issues)


def _mutations(fn):
    return [i for i in _issues(fn) if i.kind in ("scope_mutation", "impure_call")]


def _ambient(fn):
    return [i for i in _issues(fn) if i.kind == "ambient_read"]


# -- fresh locals ------------------------------------------------------------

def tuple_unpacked(n):
    a, b = [], []
    for i in range(n):
        a.append(i)
        b.append(i * 2)
    return sum(a) + sum(b)


def view_of_a_local(n, mask):
    u = np.zeros((n, n))
    inner = u[1:-1, 1:-1]
    inner[mask] = 1.0
    return u


def pipeline(path, products):
    df = read_export(path)
    df = df[df["qty"] < 500]
    df = df.merge(products, on="id")
    df["revenue"] = df["qty"] * 2
    return df


@pytest.mark.parametrize("fn", [tuple_unpacked, view_of_a_local, pipeline],
                         ids=["r17s2-tuple-unpack", "r17s4-local-view", "r17s1-pipeline"])
def test_mutating_what_the_function_made_is_not_flagged(fn):
    assert not _mutations(fn), _mutations(fn)


def mutates_a_parameter(df):
    df["x"] = 1
    return df


def mutates_a_view_of_a_parameter(arr):
    inner = arr[1:-1]
    inner[0] = 5
    return arr


def mutates_a_helpers_result(path):
    df = read_export(path)
    df["x"] = 1                     # before any copy: may be someone else's
    return df


def mutates_what_fit_returned(model, x):
    model = model.fit(x)            # sklearn returns self
    model.coef_ = 0
    return model


def mutates_an_element_of_a_fresh_dict():
    d = {"k": GLOBAL_ROWS}
    d["k"].append(1)
    return d


def rebinds_in_a_loop(groups):
    x = []
    for x in groups:
        x.append(1)                 # the caller's lists
    return groups


@pytest.mark.parametrize("fn", [
    mutates_a_parameter, mutates_a_view_of_a_parameter, mutates_a_helpers_result,
    mutates_what_fit_returned, mutates_an_element_of_a_fresh_dict, rebinds_in_a_loop,
])
def test_mutating_what_someone_else_holds_still_warns(fn):
    """The controls. `rebinds_in_a_loop` was silent BEFORE this change too:
    the old rule ignored for-loop bindings, so `x = []` made `x` fresh."""
    assert _mutations(fn), f"{fn.__name__} went quiet"


# -- ambient reads that only reach a log line ---------------------------------

def timed_in_a_print(files):
    for p in files:
        t = time.perf_counter()
        print(f"{p} {time.perf_counter() - t:.2f}s", file=sys.stderr)
    return len(files)


def timed_in_a_logger(x, logger):
    t0 = time.monotonic()
    elapsed = time.monotonic() - t0
    logger.info("took %.2f", elapsed)
    return x


def returns_the_clock():
    return time.time()


def logs_and_returns_the_clock():
    t = time.time()
    print(t)
    return t


def the_clock_decides(x):
    if time.time() > 0:
        return x
    return None


def test_a_timer_only_logged_is_not_an_ambient_read():
    assert not _ambient(timed_in_a_print)
    assert not _ambient(timed_in_a_logger)


@pytest.mark.parametrize("fn", [returns_the_clock, logs_and_returns_the_clock,
                                the_clock_decides])
def test_a_clock_that_reaches_the_result_still_warns(fn):
    assert _ambient(fn), f"{fn.__name__} went quiet"


# -- where the finding is -----------------------------------------------------

def test_lines_are_file_lines_and_the_file_is_named():
    """Lines counted from the decorator sent three testers to the wrong line."""
    issue = _mutations(mutates_a_parameter)[0]
    lines, first = inspect.getsourcelines(mutates_a_parameter)
    expected = first + next(i for i, ln in enumerate(lines) if 'df["x"] = 1' in ln)
    assert issue.line == expected
    assert os.path.basename(issue.filename) == os.path.basename(__file__)


def test_the_warning_names_the_defining_file(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    cached = c.cache(mutates_a_parameter)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cached({"x": 0})
    text = "\n".join(str(w.message) for w in rec if "IMPURE-SIDE-EFFECTS" in str(w.message))
    assert os.path.basename(__file__) in text, text
    assert "changes the function's key once" in text


# -- printed once -------------------------------------------------------------

def test_key_opaque_callable_prints_once_in_a_plain_script(tmp_path):
    """It went to logging's last-resort handler AND to warnings: twice."""
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent("""
        import cash

        class Opaque:
            __call__ = staticmethod(abs)   # no Python code to hash

        @cash.cache
        def f(cb, x):
            return cb(x)

        f(Opaque(), -5)
    """), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / ".cash")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         env=env, encoding="utf-8", errors="replace")
    assert out.stderr.count("[KEY-OPAQUE-CALLABLE]") == 1, out.stderr


def test_a_configured_log_still_gets_it():
    """The other half: a log-only reader keeps the record."""
    import logging

    from cash.diagnostics import log_diagnostic
    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("cash.test_log_diagnostic")
    handler = Grab()
    log.addHandler(handler)
    try:
        log_diagnostic(log, "KEY-OPAQUE-CALLABLE", "what", "fix")
    finally:
        log.removeHandler(handler)
    assert records and "[KEY-OPAQUE-CALLABLE]" in records[0]
