"""A ``@cash.cache`` method of an argument's class is judged by the code it wraps.

A cached function that picks code by a run-time name is stored only on a
trace, and the strict check of what the call ran leaves out the code the key
folded from the arguments (a method of an argument's class). A method of
that class that is itself ``@cash.cache``-decorated was found on the class as
the cash wrapper, whose own code is not the method's: the check judged it
anyway, and its findings (``self._validate()``, whose return it discards)
kept the outer call from ever being stored (reported against
imbalanced-learn's decorator walk, ``SMOTE._fit_resample``).

The outer call is now stored and hit in a new process; an edit to the
wrapped method still recomputes. The inner call, which changes ``self`` and
so is never stored, misses in the next process as "not stored", not as a
change to code it never read.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.timeout(300),
    pytest.mark.skipif(sys.version_info < (3, 12), reason="a trace of what a call ran needs sys.monitoring (3.12+)"),
]

OPS = textwrap.dedent("""
    def double(xs):
        return [2 * x for x in xs]
""")

BASE = textwrap.dedent("""
    import cash

    from . import ops


    class Sampler:
        def __init__(self, kind):
            self.kind = kind

        def _validate(self):
            self.checked_ = True
            return self

        @cash.cache
        def _fit_resample(self, xs):
            self._validate()
            return [x + {STEP} for x in xs]

        def fit_resample(self, xs):
            return getattr(ops, self.kind)(self._fit_resample(xs))
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from smpkg.base import Sampler

    @cash.cache
    def resample(est, xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return type(est)(est.kind).fit_resample(xs)

    print(resample(Sampler("double"), [1, 2]))
""")

INNER = textwrap.dedent("""
    from smpkg.base import Sampler

    print(Sampler("double")._fit_resample([1, 2]))
""")


def _write(tmp_path, step=1):
    pkg = tmp_path / "smpkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ops.py").write_text(OPS, encoding="utf-8")
    (pkg / "base.py").write_text(BASE.replace("{STEP}", str(step)), encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "inner.py").write_text(INNER, encoding="utf-8")


def _run(tmp_path, script="main.py"):
    proc = run_python(script, cwd=tmp_path, env={"CASH_TRACE": "1", "CASH_SUMMARY": "1"})
    return proc.stdout.strip().splitlines()[0], proc.stderr


def test_the_outer_call_is_stored_and_hits_in_a_new_process(tmp_path):
    _write(tmp_path)
    out, err = _run(tmp_path)
    assert out == "[4, 6]" and "[RUN]" in err
    assert "it picks code by a run-time name and" not in err, err
    out, err = _run(tmp_path)
    assert out == "[4, 6]"
    assert "[RUN]" not in err, f"an unedited run must hit:\n{err}"


def test_an_edit_to_the_wrapped_method_recomputes(tmp_path):
    _write(tmp_path)
    _run(tmp_path)
    _write(tmp_path, step=5)
    out, err = _run(tmp_path)
    assert out == "[12, 14]", f"served the result from before the edit: {out}\n{err}"
    assert "[RUN]" in err


def test_a_call_never_stored_misses_as_not_stored_in_the_next_process(tmp_path):
    _write(tmp_path)
    out, err = _run(tmp_path, "inner.py")
    assert out == "[2, 3]"
    assert "not stored (1x): it changed its argument 'self' in place" in err, err
    out, err = _run(tmp_path, "inner.py")
    assert out == "[2, 3]"
    assert "missed: 1 not stored last time" in err, err
