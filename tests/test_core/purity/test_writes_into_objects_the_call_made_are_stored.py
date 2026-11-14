"""A call traced for what it ran is stored when it only writes into objects it
made itself.

Found walking pvlib: a cached ``modelchain_year`` reached its helpers by a
run-time name, so what it ran was recorded, and every function that ran was
judged on its own. ``v_oc = _lambertw_v_from_i(...); v_oc[mask] = 0`` (a
helper's result, a new array) and ``longitude_obliquity_nutation(..., out)``
(``out`` an ``np.empty`` its caller made) read as side effects, and so did a
generator the call made and ran itself ("resumed a generator started before
it"). The call was never stored.

Each case runs in fresh interpreters on one cache. The writes into what the
call made store and hit; a write into the argument (``np.asarray`` hands back
the array itself, a slice is a view of it, an out-argument that is the
caller's own array) and a generator started before the call still run every
time, and so does an out-argument written in the call's own array at one
site and handed the caller's at another, through a table or a library.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

#: ``run`` reaches its helper by a run-time name (``getattr(helpers, NAME)``),
#: as ``modelchain_year`` did: judged on what its trace recorded it ran.
needs_trace = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="what a call ran is recorded with sys.monitoring, Python 3.12+; below it a call that picks code "
    "by a run-time name runs every time, with a warning",
)

pytest.importorskip("numpy")

HELPERS = textwrap.dedent("""
    import numpy as np

    def _new(n):
        v = np.full(n, np.nan)
        v[0] = 1.0
        if n == 1:
            return v.item()
        return v

    def from_helper(x):
        v = _new(len(x))
        v[v != v] = 0.0
        return v + x

    def _fill(x, out):
        out[0] = x[0] * 2
        out[1] = x[-1] * 2

    def out_argument(x):
        buf = np.empty(2)
        _fill(x, buf)
        return buf

    def copies(x):
        y = np.asarray(x, dtype=float).copy()
        y[0] = 5.0
        z = np.zeros(len(x))
        z[...] = x
        return y + z

    def _gen(n):
        for i in range(n):
            yield i * 2

    def own_generator(x):
        g = _gen(3)
        s = next(g)
        for v in g:
            s += v
        return x + s

    def through_asarray(x):
        y = np.asarray(x)
        y[0] = 9.0
        return y

    def through_a_view(x):
        y = x[1:]
        y[0] = 9.0
        return x

    def out_is_the_argument(x):
        _fill(x, x)
        return x

    _TABLE = {"fill": _fill}

    def also_through_a_table(x):
        buf = np.empty(2)
        _fill(x, buf)
        _TABLE["fill"](x, x)
        return buf

    def also_through_map(x):
        buf = np.empty(2)
        _fill(x, buf)
        list(map(_fill, [x], [x]))
        return buf

    _STARTED = _gen(100)
    next(_STARTED)

    def started_before(x):
        return x + next(_STARTED)
""")

MAIN = textwrap.dedent("""
    import sys
    import numpy as np
    import cash
    import helpers

    NAME = sys.argv[1]

    @cash.cache
    def run(n):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        x = np.arange(n, dtype=float)
        return float(np.sum(getattr(helpers, NAME)(x)))

    print(run(4))
""")


def _run(tmp_path, name):
    proc = run_python("main.py", name, cwd=tmp_path, env={"PYTHONWARNINGS": "always"})
    return proc.stdout.strip(), proc.stderr


@pytest.fixture
def project(tmp_path):
    (tmp_path / "helpers.py").write_text(HELPERS, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    return tmp_path


@needs_trace
@pytest.mark.parametrize("name", ["from_helper", "out_argument", "copies", "own_generator"])
def test_writes_into_what_the_call_made_store_and_hit(project, name):
    first, err = _run(project, name)
    assert "[RUN]" in err
    again, err = _run(project, name)
    assert again == first
    assert "[RUN]" not in err, f"{name}: the next process ran the call again:\n{err[-1500:]}"


@needs_trace
@pytest.mark.parametrize(
    "name, why",
    [
        ("through_asarray", "y[...] = ..."),
        ("through_a_view", "y[...] = ..."),
        ("out_is_the_argument", "changes the argument 'out' in place"),
        ("also_through_a_table", "changes the argument 'out' in place"),
        ("also_through_map", "changes the argument 'out' in place"),
        ("started_before", "resumed a generator started before it"),
    ],
)
def test_writes_into_what_existed_before_still_run_every_time(project, name, why):
    _, err = _run(project, name)
    assert "[RUN]" in err and why in err, err[-1500:]
    _, err = _run(project, name)
    assert "[RUN]" in err, f"{name}: served from the cache:\n{err[-1500:]}"


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="below Python 3.12 nothing records what a call ran")
@pytest.mark.parametrize("name", ["from_helper", "through_asarray"])
def test_without_a_trace_the_call_runs_every_time_and_says_why(project, name):
    for _ in range(2):
        _, err = _run(project, name)
        assert "[RUN]" in err and "KEY-UNTRACKABLE-DEP" in err and "main.run" in err, err[-1500:]
