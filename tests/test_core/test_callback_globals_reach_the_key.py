"""Globals read by code passed as an argument reach the key.

This turned a physics-breaking commit into a green CI run:

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

Each shape it takes: a plain function, a bound method, and a
callable instance (whose code lives on its class). Fresh process per run.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

POTENTIALS = textwrap.dedent("""
    TILT = {tilt}

    def double_well(x):
        return x ** 4 - x ** 2 + TILT * x

    class Well:
        def force(self, x):
            return TILT * x

    class Callable:
        def __call__(self, x):
            return TILT * x * 2
""")

MAIN = textwrap.dedent("""
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
""")


def _project(tmp_path, tilt):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "potentials.py").write_text(POTENTIALS.format(tilt=tilt), encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    return proj


def _run(tmp_path, proj):
    out = run_python("main.py", cwd=proj, cache_dir=tmp_path / "cache")
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr.count("RAN")


def _oracle(tilt):
    xs = [i / 100 for i in range(100)]
    return [
        round(sum(x**4 - x**2 + tilt * x for x in xs), 6),
        round(sum(tilt * x for x in xs), 6),
        round(sum(tilt * x * 2 for x in xs), 6),
    ]


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


SMOOTHING = textwrap.dedent("""
    import functools

    T = 1

    def scale(x, k):
        return x * k

    SMOOTH = functools.partial(scale, k=2)

    def callback(x):
        return SMOOTH(x)

    class Instance:
        def __call__(self, x):
            return SMOOTH(x)

    def helper(x):
        return x * T

    def via_helper(x):
        return helper(x)
""")


@pytest.fixture
def smoothing(tmp_path, monkeypatch):
    (tmp_path / "smoothing_cb.py").write_text(SMOOTHING, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("smoothing_cb", None)
    import smoothing_cb

    yield smoothing_cb
    sys.modules.pop("smoothing_cb", None)


def test_what_a_callbacks_helpers_and_bindings_carry_reaches_the_key(smoothing, cash_instance):
    """A callback is keyed by what the cached function's own reads are keyed
    by: a global partial it calls, and a global its helper reads."""

    @cash_instance.cache
    def run(cb, x):
        return cb(x)

    instance = smoothing.Instance()
    assert [run(smoothing.callback, 1), run(instance, 1), run(smoothing.via_helper, 1)] == [2, 2, 1]
    assert [run(smoothing.callback, 1), run(instance, 1), run(smoothing.via_helper, 1)] == [2, 2, 1]
    assert run.cache_info()["hits"] == 3

    smoothing.SMOOTH = smoothing.functools.partial(smoothing.scale, k=3)
    smoothing.T = 5
    assert [run(smoothing.callback, 1), run(instance, 1), run(smoothing.via_helper, 1)] == [3, 3, 5]
