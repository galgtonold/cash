"""A function default keys the globals it reads, and a function attribute is data.

``def g(x, fn=lambda v: v + K)`` evaluates the lambda where the ``def``
stands, so ``K`` is in no scope of ``g``'s: the lambda was keyed by its code
only, and editing ``K`` served the old result. The same for data stored as an
attribute of a function (``scale.k = 1``) and read as ``scale.k``, which the
function's source does not show.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

HELPER = textwrap.dedent("""
    K = {k}

    def g(x, fn=lambda v: v + K):
        return fn(x)
""")

MAIN = textwrap.dedent("""
    import json, sys, time, warnings
    warnings.simplefilter("ignore")
    import cash, helper

    @cash.cache
    def f(x):
        print("RAN", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)
        return helper.g(x)

    print(json.dumps(f(1)))
""")


def _run(tmp_path, k):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "helper.py").write_text(HELPER.format(k=k), encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    env = {n: v for n, v in os.environ.items() if not n.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out = subprocess.run([sys.executable, "main.py"], cwd=str(proj), capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr.count("RAN")


def test_editing_what_a_default_lambda_reads_recomputes(tmp_path):
    """THE BUG: K = 1 -> 100 served 2 where an uncached run gives 101."""
    assert _run(tmp_path, 1) == (2, 1)
    assert _run(tmp_path, 1) == (2, 0)
    assert _run(tmp_path, 100) == (101, 1)


def _scale(v):
    return v


_scale.k = 1


def _helper(x):
    return x + _scale.k


def test_a_function_attribute_read_by_a_helper_is_keyed(monkeypatch):
    """`scale.k` changed between two calls of one process."""
    import cash

    c = cash.Cash(backend=cash.InMemoryBackend())

    @c.cache
    def f(x):
        return _helper(x)

    monkeypatch.setattr(_scale, "k", 1)
    assert f(1) == 2
    monkeypatch.setattr(_scale, "k", 7)
    assert f(1) == 8
