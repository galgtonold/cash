"""An argument whose methods import modules lazily is cached on a trace.

Reported against imbalanced-learn's commit walk: a sampler passed to a cached
function has methods that import modules not loaded yet (``from ._tags
import get_tags`` inside a method) and pick code by a run-time name
(``getattr(sparse, name)(steps)``). Once code an argument carries was
checked for run-time lookups, the lazy imports, whose module's top level
cash cannot read without running it, refused every call on every Python,
and the lookups by name were not served either: what the trace saw run
(a ``@dataclass`` ``__init__``, ``getattr(x, "name", None)``, a base class's
``__init__`` setting up an instance the call made) left it unusable.

On Python 3.12 and newer the call is served on a trace of what it ran, and
an edit to a lazily imported module that ran, or to the function the name
picks, recomputes. With no trace (``CASH_TRACE=0``, Python before 3.12) an
edit to a module imported lazily cannot be seen, so such a call runs every
time.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

TRACED = sys.version_info >= (3, 12)

OPS = textwrap.dedent("""
    def double(xs):
        return [{DOUBLE} * x for x in xs]


    def triple(xs):
        return [3 * x for x in xs]
""")

# Not readable without running it: a call cash does not know at the top level.
TAGS = textwrap.dedent("""
    import dataclasses
    import json

    FACTOR = json.loads("{FACTOR}")


    @dataclasses.dataclass
    class Tags:
        factor: int = 1


    def get_tags(est):
        return Tags(factor=FACTOR)
""")

# The same without a dataclass: the value reaches the result only through a
# local the method binds, which no global read names.
CONSTANTS = textwrap.dedent("""
    import json

    OFFSET = json.loads("{OFFSET}")
""")

LATE = textwrap.dedent("""
    import json

    EXTRA = json.loads("7")
""")

BASE = textwrap.dedent("""
    from . import ops


    class BaseSampler:
        def __init__(self, kind):
            self.kind = kind


    class Sampler(BaseSampler):
        def __init__(self, kind):
            super().__init__(kind)

        def fit_resample(self, xs):
            from ._tags import get_tags
            from ._constants import OFFSET

            columns = getattr(xs, "columns", None)
            factor = get_tags(self).factor
            return [v * factor + OFFSET for v in self._make_samples(xs)], columns

        def _make_samples(self, xs):
            func = self.kind
            return getattr(ops, func)(xs)

        def _more_tags(self):
            from ._late import EXTRA

            return EXTRA
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from imbpkg.base import Sampler

    @cash.cache
    def resample(est, xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return type(est)(est.kind).fit_resample(xs)

    print(resample(Sampler("double"), [1, 2])[0])
""")


def _write(tmp_path, *, double=2, factor=3, offset=0):
    pkg = tmp_path / "imbpkg"
    pkg.mkdir(exist_ok=True)
    files = {
        "__init__.py": "",
        "ops.py": OPS.replace("{DOUBLE}", str(double)),
        "_tags.py": TAGS.replace("{FACTOR}", str(factor)),
        "_constants.py": CONSTANTS.replace("{OFFSET}", str(offset)),
        "_late.py": LATE,
        "base.py": BASE,
    }
    for name, text in files.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")


def _run(tmp_path, trace="1"):
    proc = run_python("main.py", cwd=tmp_path, env={"CASH_TRACE": trace, "CASH_SUMMARY": "1"})
    return proc.stdout.strip().splitlines()[0], proc.stderr


@pytest.mark.skipif(not TRACED, reason="a trace of what a call ran needs sys.monitoring (Python 3.12+)")
def test_a_second_process_hits_on_the_trace(tmp_path):
    _write(tmp_path)
    out, err = _run(tmp_path)
    assert out == "[6, 12]" and "[RUN]" in err
    out, err = _run(tmp_path)
    assert out == "[6, 12]"
    assert "[RUN]" not in err, f"an unedited run must hit:\n{err}"


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize(
    "edit, want",
    [
        ({"factor": 5}, "[10, 20]"),
        ({"offset": 100}, "[106, 112]"),
        ({"double": 20}, "[60, 120]"),
    ],
    ids=["lazy_module_global_read", "lazy_module_value_bound_to_a_local", "function_getattr_picks"],
)
def test_an_edit_to_what_the_call_ran_recomputes(tmp_path, edit, want, trace):
    _write(tmp_path)
    out, _ = _run(tmp_path, trace)
    assert out == "[6, 12]"
    out, _ = _run(tmp_path, trace)
    assert out == "[6, 12]"
    _write(tmp_path, **edit)
    out, err = _run(tmp_path, trace)
    assert out == want, f"served the result from before the edit: {out}\n{err}"


def test_without_a_trace_a_lazy_import_runs_uncached(tmp_path):
    """With no trace, an edit to a module the call imports lazily cannot be
    seen: the call runs every time, and the miss names the import (before
    3.12 the lookup by name refuses it too, and is named first)."""
    _write(tmp_path)
    _run(tmp_path, "0")
    out, err = _run(tmp_path, "0")
    assert out == "[6, 12]"
    assert "[RUN]" in err, "a call no trace checks must not be served"
    label = "imports a module cash cannot read before it is loaded" if TRACED else "code picked by a run-time name"
    assert f"missed: 1 {label}" in err, err


# Once the lazily imported module is loaded, the walk can key it; the
# argument's findings are looked at again, so the next call is cached even
# with no trace.
SAME_PROCESS = textwrap.dedent("""
    import sys
    import cash

    class Sampler:
        def fit_resample(self, xs):
            from imbpkg._constants import OFFSET

            return [x + OFFSET for x in xs]

    @cash.cache
    def resample(est, xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return est.fit_resample(xs)

    print(resample(Sampler(), [1]))
    print(resample(Sampler(), [2]))
    print(resample(Sampler(), [2]))
""")


@pytest.mark.parametrize("trace", ["1", "0"])
def test_once_the_module_is_loaded_the_argument_is_looked_at_again(tmp_path, trace):
    _write(tmp_path)
    (tmp_path / "main.py").write_text(SAME_PROCESS, encoding="utf-8")
    proc = run_python("main.py", cwd=tmp_path, env={"CASH_TRACE": trace})
    assert proc.stdout.split() == ["[1]", "[2]", "[2]"]
    assert proc.stderr.count("[RUN]") == 2, proc.stderr


# Reported next, on the same walk: ``Pipeline.predict`` sits behind sklearn's
# ``available_if``, a descriptor that keeps the function as ``fn``, and
# ``getattr(y, "dtype", None)`` finds a slot of a C type. Neither could be
# checked again, so the call was not stored.
AVAILABLE = textwrap.dedent("""
    class _AvailableIf:
        def __init__(self, fn, check):
            self.fn = fn
            self.check = check

        def __get__(self, obj, owner=None):
            if obj is None:
                return self
            self.check(obj)
            return self.fn.__get__(obj, owner)


    def available_if(check):
        return lambda fn: _AvailableIf(fn, check)
""")

PIPELINE = textwrap.dedent("""
    from . import ops
    from ._available import available_if


    def _has_kind(pipe):
        return bool(pipe.kind)


    class Pipeline:
        def __init__(self, kind):
            self.kind = kind

        @available_if(_has_kind)
        def predict(self, xs):
            imag = getattr(float(xs[0]), "imag", None)
            return [v + imag for v in getattr(ops, self.kind)(xs)]
""")

PIPELINE_MAIN = textwrap.dedent("""
    import sys
    import cash
    from imbpkg.pipeline import Pipeline

    @cash.cache
    def predict(pipe, xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return pipe.predict(xs)

    print(predict(Pipeline("double"), [1, 2]))
""")


@pytest.mark.skipif(not TRACED, reason="a trace of what a call ran needs sys.monitoring (Python 3.12+)")
def test_a_method_behind_a_descriptor_and_a_c_slot_lookup_hit_on_the_trace(tmp_path):
    _write(tmp_path)
    (tmp_path / "imbpkg" / "_available.py").write_text(AVAILABLE, encoding="utf-8")
    (tmp_path / "imbpkg" / "pipeline.py").write_text(PIPELINE, encoding="utf-8")
    (tmp_path / "main.py").write_text(PIPELINE_MAIN, encoding="utf-8")
    out, err = _run(tmp_path)
    assert out == "[2.0, 4.0]" and "[RUN]" in err
    out, err = _run(tmp_path)
    assert out == "[2.0, 4.0]"
    assert "[RUN]" not in err, f"an unedited run must hit:\n{err}"
    _write(tmp_path, double=20)
    (tmp_path / "main.py").write_text(PIPELINE_MAIN, encoding="utf-8")
    out, _ = _run(tmp_path)
    assert out == "[20.0, 40.0]", f"served the result from before the edit: {out}"
