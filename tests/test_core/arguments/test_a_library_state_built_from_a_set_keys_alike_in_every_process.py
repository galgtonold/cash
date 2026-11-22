"""A library's object whose own reduce builds its state from a set keys
alike in every process.

lmfit's ``Parameters`` (a dict subclass) saves ``{key: ... for key in
interpreter.user_defined_symbols()}``, a set, and holds an asteval
interpreter whose symbol table loops back to it; uncertainties'
``AffineScalarFunc.__getstate__`` fills its dict from a set of slot names.
The interpreter, which the class's reduce leaves out, made a
``Parameters``, a ``Minimizer`` or a ``MinimizerResult`` argument
unkeyable (KEY-UNHASHABLE-ARG), and the order PYTHONHASHSEED gave each set
keyed a fit result apart in every process: a call handed one missed in a
second process. The library below has the same shapes, without lmfit.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from tests._scripts import child_env, run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(240)]

LIBRARY = """
    import math

    NAMES = ("pi", "e", "tau", "inf", "sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceil")


    class Interpreter:
        def __init__(self):
            self.symtable = {name: getattr(math, name) for name in NAMES}
            self.builtins = set(self.symtable)
            self.symtable["interpreter"] = self

        def user_defined(self):
            return set(self.symtable) - self.builtins - {"interpreter"}


    class Param:
        def __init__(self, name, value):
            self.name = name
            self.value = value


    class Params(dict):
        def __init__(self):
            super().__init__()
            self._interp = Interpreter()

        def add(self, name, value):
            self[name] = Param(name, value)
            self._interp.symtable[name] = value

        def __reduce__(self):
            symbols = {key: self._interp.symtable[key] for key in self._interp.user_defined()}
            return self.__class__, (), {"symbols": symbols, "params": [self[k] for k in self]}

        def __setstate__(self, state):
            for p in state["params"]:
                self.add(p.name, p.value)


    class Uval:
        __slots__ = ("nominal", "std", "tag", "low", "high", "unit")

        def __init__(self, nominal, std):
            self.nominal, self.std = nominal, std
            self.tag, self.low, self.high, self.unit = "x", -1.0, 1.0, "m"

        def __getstate__(self):
            return {name: getattr(self, name) for name in {"nominal", "std", "tag", "low", "high", "unit"}}

        def __setstate__(self, state):
            for k, v in state.items():
                setattr(self, k, v)


    class Fit:
        def __init__(self, params):
            self.params = params
            self.uvars = {name: Uval(p.value, 0.1) for name, p in params.items()}
"""

MAIN = """
    import sys
    import warnings

    import cash
    from fitlib import Fit, Params, Uval


    @cash.cache
    def total(params, fit, u):
        return sum(p.value for p in params.values()) + sum(v.nominal for v in fit.uvars.values()) + u.nominal


    params = Params()
    for i, name in enumerate("abcdefgh"):
        params.add(name, float(i if sys.argv[1] == "same" else i + 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = [total(params, Fit(params), Uval(2.0, 0.5)) for _ in range(2)]
    info = total.cache_info()
    # Below Python 3.12 the first call a process stores says this once.
    unseen = "KEY-LIBRARY-STATE-UNSEEN" if sys.version_info < (3, 12) else ""
    codes = sorted({str(getattr(w.message, "code", "")) for w in caught} - {"", unseen})
    print(out, info["hits"], info["misses"], codes)
"""


def _setup(tmp_path: Path) -> tuple[Path, dict]:
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "fitlib.py").write_text(textwrap.dedent(LIBRARY), encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text(textwrap.dedent(MAIN), encoding="utf-8")
    return project, {"PYTHONPATH": f"{child_env()['PYTHONPATH']}{os.pathsep}{site}"}


def _run(project: Path, env: dict, which: str, seed: str) -> str:
    return (
        run_python("main.py", which, cwd=project, env={**env, "PYTHONHASHSEED": seed}).stdout.strip().splitlines()[-1]
    )


def test_a_fit_like_argument_keys_alike_under_another_hash_seed(tmp_path):
    project, env = _setup(tmp_path)
    assert _run(project, env, "same", "1") == "[58.0, 58.0] 1 1 []"
    # Another process, its sets in another order: the same call.
    assert _run(project, env, "same", "2") == "[58.0, 58.0] 2 0 []"
    assert _run(project, env, "same", "3") == "[58.0, 58.0] 2 0 []"
    # Another value is another call.
    assert _run(project, env, "other", "2") == "[74.0, 74.0] 1 1 []"


STORED = """
    from cash import kept_state
    from fitlib import Uval

    value = Uval(2.0, 0.5)
    state = value.__getstate__()
    _, also_keyed = kept_state.dumps_also_keyed(value)
    print(list(state) == sorted(state), also_keyed)
"""


def test_its_store_bytes_are_not_taken_for_its_key_bytes(tmp_path):
    """A store pickle of such an object writes its state in the set's
    order, which its key puts in name order: those bytes are not its key's
    too. Taken for them, a fitted scikit-learn estimator a notebook
    variable held digested apart from its key, read as changed by a
    statement that did not touch it, and refused that statement's store."""
    project, env = _setup(tmp_path)
    (project / "stored.py").write_text(textwrap.dedent(STORED), encoding="utf-8")
    for seed in ("1", "2", "3", "4"):
        out = run_python("stored.py", cwd=project, env={**env, "PYTHONHASHSEED": seed}).stdout.split()
        assert out[-1] == ("True" if out[-2] == "True" else "False"), (seed, out)
