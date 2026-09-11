"""A cached function handed around as a value is a dependency like one that is called.

Round 19 (r19s5): ``def outer(n): return sum(map(inner, [n]))`` with ``inner``
cached -- editing ``inner``'s helper recomputed ``inner`` but ``outer`` HIT its
old sum. The same held for ``pool.map(inner, ...)``, ``delayed(inner)``, a
``for fn in [inner]`` loop, ``fn=inner`` as a default, and a module-level
``partial(inner)``: only a CALL written in the body made a graph edge. That is
the ordinary way to parallelise a cached step.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

HELPERS = "def helper(i):\n    return i * {K}\n"

INNER = '''\
import cash
from helpers import helper


@cash.cache
def inner(n):
    return sum(helper(i) for i in range(n))
'''

JOB = '''\
import functools, sys, time
import cash
from inner import inner

P = functools.partial(inner)


def run(tag):
    print("[RUN]", tag, file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe


@cash.cache
def by_map(n):
    run("map")
    return sum(map(inner, [n]))


@cash.cache
def by_partial_global(n):
    run("partial-global")
    return P(n)


@cash.cache
def by_list(n):
    run("list")
    return sum(fn(n) for fn in [inner])


@cash.cache
def by_default(n, fn=inner):
    run("default")
    return fn(n)


@cash.cache
def by_call(n):
    run("call")
    return inner(n)


print(by_map(10), by_partial_global(10), by_list(10), by_default(10), by_call(10))
'''

FORMS = {"map", "partial-global", "list", "default", "call"}


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    ran = {line.split(maxsplit=1)[1] for line in p.stderr.splitlines()
           if line.startswith("[RUN] ")}
    return p.stdout.strip(), ran


def test_editing_a_cached_function_passed_as_a_value_invalidates_its_user(tmp_path):
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    (tmp_path / "inner.py").write_text(INNER, encoding="utf-8")
    (tmp_path / "helpers.py").write_text(HELPERS.format(K=2), encoding="utf-8")

    assert _run(tmp_path) == ("90 90 90 90 90", FORMS)
    assert _run(tmp_path) == ("90 90 90 90 90", set()), "an unedited run did not hit"
    (tmp_path / "helpers.py").write_text(HELPERS.format(K=3), encoding="utf-8")
    assert _run(tmp_path) == ("135 135 135 135 135", FORMS), "an old sum was served"


def test_a_referenced_cached_function_is_an_edge_but_an_attribute_of_an_instance_is_not_read():
    """The reference resolver goes through modules and classes only: analysis
    must not run a property to find out what a name holds."""
    from cash.notebook.analysis import CodeAnalyzer

    class Loud:
        @property
        def fn(self):
            raise RuntimeError("a property ran during analysis")

    ns = {"obj": Loud()}
    assert CodeAnalyzer._referenced_function("obj.fn", ns) is None
