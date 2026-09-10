"""Globals read by code passed as an argument reach the key.

CAS-113, round-17 tester r17s4 -- their most important finding, because it
turned a physics-breaking commit into a green CI run:

    # potentials.py
    TILT = 0.0
    def double_well(x): return x**4 - x**2 + TILT * x

    @cash.cache
    def integrate(force, n): ...        # force passed as an ARGUMENT
    integrate(potentials.double_well, 10_000)

Change TILT and the old result came back, silently. Code-as-argument hashed
the callback's CODE but never ran the global-read fold over it; the fold ran
only over the cached function's own body and its followed helpers, and an
argument is neither. A helper one level below the callback WAS folded, which
is what made the gap easy to trust.

Each shape the tester found: a plain function, a bound method, and a
callable instance (whose code lives on its class). Fresh process per run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

POTENTIALS = textwrap.dedent('''
    TILT = {tilt}

    def double_well(x):
        return x ** 4 - x ** 2 + TILT * x

    class Well:
        def force(self, x):
            return TILT * x

    class Callable:
        def __call__(self, x):
            return TILT * x * 2
''')

MAIN = textwrap.dedent('''
    import json, sys, time
    import cash
    import potentials

    @cash.cache
    def integrate(force, n):
        print("RAN", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)
        return round(sum(force(i / n) for i in range(n)), 6)

    print(json.dumps([
        integrate(potentials.double_well, 100),
        integrate(potentials.Well().force, 100),
        integrate(potentials.Callable(), 100),
    ]))
''')


def _project(tmp_path, tilt):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "potentials.py").write_text(POTENTIALS.format(tilt=tilt), encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    return proj


def _run(tmp_path, proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out = subprocess.run([sys.executable, "main.py"], cwd=str(proj),
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr.count("RAN")


def _oracle(tilt):
    xs = [i / 100 for i in range(100)]
    return [round(sum(x ** 4 - x ** 2 + tilt * x for x in xs), 6),
            round(sum(tilt * x for x in xs), 6),
            round(sum(tilt * x * 2 for x in xs), 6)]


def test_every_callback_shape_sees_the_constant_change(tmp_path):
    """THE BUG: function, bound method and callable instance all served stale."""
    proj = _project(tmp_path, tilt=0.0)
    first, _ = _run(tmp_path, proj)
    assert first == _oracle(0.0)

    _project(tmp_path, tilt=0.5)
    second, ran = _run(tmp_path, proj)

    assert second == _oracle(0.5), f"stale callback results: {second}"
    assert ran == 3


def test_an_unchanged_constant_still_hits(tmp_path):
    """The control that matters: folding the callback's globals costs no hit."""
    proj = _project(tmp_path, tilt=0.0)
    _run(tmp_path, proj)
    again, ran = _run(tmp_path, proj)

    assert again == _oracle(0.0)
    assert ran == 0, f"{ran} callback call(s) recomputed with nothing changed"
