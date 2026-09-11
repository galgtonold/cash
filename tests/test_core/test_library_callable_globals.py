"""What a library-made callable was built with reaches the key when it is read as a global.

Round 19 (r19s2): ``SMOOTHER = partial(ndimage.gaussian_filter, sigma=SIGMA)``
in a config module, called by a cached function -- SIGMA 1.0 -> 2.0 was a HIT
with the old histogram. So were ``np.poly1d(coeffs)`` and scipy's
``interp1d(table)``. The same partial passed as an argument was keyed, and so
was a partial over the user's OWN function (its binding is followed as a
helper); over library code nothing followed it.

Each form is its own cached function in one job, so three processes cover all
of them: cold, warm (must hit -- a key that moves every run would pass the
edit check too), and after the edit.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

CFG = '''\
from functools import partial
import numpy as np

K = {K}
ROUND = partial(round, ndigits=K)          # a partial over a builtin
POLY = np.poly1d([K, 0.0, 1.0])            # a library callable object holding data
RATES = {{"a": K, "b": 10}}
LOOKUP = RATES.get                         # a builtin bound method of a dict
'''

JOB = '''\
import sys, time
from functools import partial
import numpy as np
import cash
import cfgmod
from cfgmod import ROUND, POLY, LOOKUP

SAME = partial(round, ndigits=cfgmod.K)    # built in the job's own module


def run(tag):
    print("[RUN]", tag, file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe


@cash.cache
def by_from_import(x):
    run("from-import")
    return ROUND(x)


@cash.cache
def by_module_attr(x):
    run("module-attr")
    return cfgmod.ROUND(x)


@cash.cache
def by_poly(x):
    run("poly1d")
    return float(POLY(x))


@cash.cache
def by_lookup(key):
    run("bound-method")
    return LOOKUP(key)


@cash.cache
def by_same_module(x):
    run("same-module")
    return SAME(x)


print(by_from_import(3.14159), by_module_attr(3.14159), by_poly(2.0),
      by_lookup("a"), by_same_module(3.14159))
'''

FORMS = {"from-import", "module-attr", "poly1d", "bound-method", "same-module"}


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    ran = {line.split(maxsplit=1)[1] for line in p.stderr.splitlines()
           if line.startswith("[RUN] ")}
    return p.stdout.strip(), ran


def test_editing_what_a_library_callable_global_was_built_with_invalidates(tmp_path):
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    (tmp_path / "cfgmod.py").write_text(CFG.format(K=2), encoding="utf-8")

    assert _run(tmp_path) == ("3.14 3.14 9.0 2 3.14", FORMS)
    assert _run(tmp_path) == ("3.14 3.14 9.0 2 3.14", set()), "an unedited run did not hit"
    (tmp_path / "cfgmod.py").write_text(CFG.format(K=3), encoding="utf-8")
    out, ran = _run(tmp_path)
    assert out == "3.142 3.142 13.0 3 3.142", "a callable's old arguments were served"
    assert ran == FORMS


STATEFUL_JOB = '''\
import sys, time
import numpy as np
import cash

DRAW = np.random.default_rng(0).normal     # advances its generator when called
SINE = np.vectorize(np.sin, otypes=[float])  # fills a cache on its first call


@cash.cache
def noisy(n):
    print("[RUN] noisy", file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe
    return round(float(DRAW(size=n).sum()), 6)


@cash.cache
def wave(x):
    print("[RUN] wave", file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe
    return float(SINE(np.asarray(x)).sum())


print(noisy(3), noisy(3), noisy(3), wave([0.5, 1.0]), wave([0.5, 1.0]), wave([0.5, 1.0]))
'''


def test_a_callable_that_changes_when_called_does_not_miss_forever(tmp_path):
    """Control: the carried state is watched like a provisional global. A
    callable whose own state moves when called (a bound generator method,
    np.vectorize's cache) stops being keyed after the first call that moved
    it -- at most one extra miss in a process, and the next process hits."""
    (tmp_path / "job.py").write_text(STATEFUL_JOB, encoding="utf-8")
    first, ran_first = _run(tmp_path)
    assert ran_first == {"noisy", "wave"}
    noisy = first.split()[:3]
    assert noisy[1] == noisy[2], "a third identical call missed again: keyed on moving state"
    second, ran_second = _run(tmp_path)
    assert ran_second == set(), "the next process missed"
