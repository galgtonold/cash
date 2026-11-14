"""A class attribute a class-body loop over a set left behind is keyed by the set.

Reported against imbalanced-learn's ``EasyEnsembleClassifier``::

    excluded_params = {"bootstrap", "oob_score", ...}
    for param in excluded_params:
        _parameter_constraints.pop(param, None)

leaves the class attribute ``param`` holding whichever string the set
iterates last, which follows ``PYTHONHASHSEED``: keyed by its value, every
process read the class as changed and missed. Which element that is, is the
set's iteration order, which no key holds for any set (a result built from
``list(some_set)`` is served whatever order a later process iterates in):
the attribute is keyed as the set's leftover, the set beside it. An edit to
the set, the attribute rebound to another value, or a name bound otherwise
than by that one loop stays keyed by value.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from cash.decorator.loop_leftovers import leftover_value, set_loop_leftovers
from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

ENSEMBLE = textwrap.dedent("""
    class Ensemble:
        constraints = {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "eps": 5, "zeta": 6}
        excluded = {EXCLUDED}
        constraints = dict(constraints)
        for param in excluded:
            constraints.pop(param, None)

        def __init__(self, n):
            self.n = n

        def fit(self, xs):
            return [x * self.n + len(self.constraints) for x in xs]
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from enspkg.ens import Ensemble

    @cash.cache
    def run(est, xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return est.fit(xs)

    print(run(Ensemble(2), [1, 2]), Ensemble.param)
""")


def _write(tmp_path, excluded='"beta", "delta", "zeta", "eps"'):
    pkg = tmp_path / "enspkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ens.py").write_text(ENSEMBLE.replace("{EXCLUDED}", "{" + excluded + "}"), encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")


def _run(tmp_path, seed, trace="1"):
    proc = run_python(
        "main.py", cwd=tmp_path, env={"PYTHONHASHSEED": str(seed), "CASH_TRACE": trace, "CASH_SUMMARY": "1"}
    )
    *result, param = proc.stdout.split()
    return " ".join(result), param, proc.stderr


@pytest.mark.parametrize("trace", ["1", "0"])
def test_processes_with_other_hash_seeds_hit(tmp_path, trace):
    _write(tmp_path)
    out, _, err = _run(tmp_path, 1, trace)
    assert out == "[4, 6]" and "[RUN]" in err
    leftovers = set()
    for seed in range(2, 9):
        out, param, err = _run(tmp_path, seed, trace)
        leftovers.add(param)
        assert out == "[4, 6]"
        assert "[RUN]" not in err, f"seed {seed} left param={param!r} and missed:\n{err}"
    assert len(leftovers) > 1, "the seeds tried must leave different elements for the test to mean anything"


def test_an_edit_to_the_set_recomputes(tmp_path):
    _write(tmp_path)
    _run(tmp_path, 1)
    _write(tmp_path, excluded='"beta", "delta", "zeta", "eps", "gamma"')
    out, _, err = _run(tmp_path, 1)
    assert out == "[3, 5]", f"served the result from before the edit: {out}\n{err}"


def test_the_rule_reads_the_class_body(tmp_path, monkeypatch):
    """Only a name the one loop binds, over a set the class bound above it;
    the value checked on every read."""
    src = tmp_path / "lmod.py"
    src.write_text(
        textwrap.dedent("""
            NAMES = {"a", "b", "c"}

            class Loop:
                names = {"a", "b", "c"}
                for name in names:
                    pass

            class Twice:
                names = {"a", "b", "c"}
                for name in names:
                    pass
                name = "a"

            class OverAGlobal:
                for name in NAMES:
                    pass
                names = NAMES

            class Constant:
                names = {"a", "b", "c"}
                name = "a"
        """),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "lmod", raising=False)
    import lmod

    names = ("names", "name")
    assert set_loop_leftovers(lmod.Loop, names) == {"name": "names"}
    for cls in (lmod.Twice, lmod.OverAGlobal, lmod.Constant):
        assert set_loop_leftovers(cls, names) == {}, cls.__name__
    ns = vars(lmod.Loop)
    leftovers = {"name": "names"}
    assert leftover_value(ns, "name", leftovers) == ("cash:last element of the set", "names")
    other = next(n for n in lmod.Loop.names if n != lmod.Loop.name)
    lmod.Loop.name = other  # rebound: keyed by value again
    assert leftover_value(vars(lmod.Loop), "name", leftovers) == other
