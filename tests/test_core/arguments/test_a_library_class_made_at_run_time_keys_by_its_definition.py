"""An argument holding a library's class or function that has no name to
be found by keys by what defines it.

specutils builds a spectrum's axis model with ``type("SpectralTabular1D",
(Tabular1D,), {...})`` inside a function; astropy's model metaclass gives
the class operator methods made in a function, and ``u.spectral()`` holds
lambdas. Pickle finds none of them by name, so every ``Spectrum``
argument ran uncached (KEY-UNHASHABLE-ARG) while a bare Quantity keyed.
A key is hashed, never loaded: such a class is keyed by its metaclass,
name, bases and attributes, such a function by its code and what it
captured, everything they hold hashed like the rest of the argument.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._scripts import child_env, run_python

pytestmark = [pytest.mark.core]

LIBRARY = """
    import threading

    import numpy as np


    def _oper(name):
        return lambda left, right: (name, left, right)


    class Meta(type):
        def __new__(mcls, name, bases, ns):
            ns.setdefault("__add__", _oper("add"))
            return super().__new__(mcls, name, bases, ns)


    class Tabular(metaclass=Meta):
        def __init__(self, points, lookup_table):
            self.points = np.asarray(points)
            self.lookup_table = np.asarray(lookup_table)


    class Spectrum:
        def __init__(self, flux, axis, unit="um", lock=False):
            factor = {"um": 1.0, "nm": 1e-3}[unit]
            Model = type(
                "AxisTabular",
                (Tabular,),
                {"equivalencies": [(unit, "Hz", lambda x: factor / x, lambda x: factor * x)]},
            )
            if lock:
                Model.guard = threading.Lock()
            self.flux = np.asarray(flux, dtype=float)
            self.model = Model(np.arange(len(axis)), axis)
            self.model.inverse = Model(axis, np.arange(len(axis)))
"""

MAIN = """
    import sys
    import warnings

    import cash
    from speclib import Spectrum

    warnings.simplefilter("always")


    @cash.cache
    def total(s):
        return float(s.flux.sum() + s.model.lookup_table[-1])


    def spectrum(flux, axis, unit="um"):
        return Spectrum(flux, axis, unit)


    calls = {
        "same": [spectrum([1, 2, 3], [1.0, 2.0, 3.0]), spectrum([1, 2, 3], [1.0, 2.0, 3.0])],
        "flux": [spectrum([1, 2, 4], [1.0, 2.0, 3.0])],
        "axis": [spectrum([1, 2, 3], [1.0, 2.0, 5.0])],
        "unit": [spectrum([1, 2, 3], [1.0, 2.0, 3.0], "nm")],
        "lock": [Spectrum([1, 2, 3], [1.0, 2.0, 3.0], lock=True)],
    }[sys.argv[1]]
    with warnings.catch_warnings(record=True) as caught:
        out = [(total(s), float(s.flux.sum() + s.model.lookup_table[-1])) for s in calls]
    info = total.cache_info()
    # Below Python 3.12 the first call a process stores says this once.
    unseen = "KEY-LIBRARY-STATE-UNSEEN" if sys.version_info < (3, 12) else ""
    codes = sorted({str(getattr(w.message, "code", "")) for w in caught} - {"", unseen})
    print(out, info["hits"], info["misses"], codes)
"""


def _setup(tmp_path: Path) -> tuple[Path, dict]:
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "speclib.py").write_text(textwrap.dedent(LIBRARY), encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text(textwrap.dedent(MAIN), encoding="utf-8")
    env = {"PYTHONPATH": f"{child_env()['PYTHONPATH']}{__import__('os').pathsep}{site}"}
    return project, env


def _run(project: Path, env: dict, which: str) -> str:
    return run_python("main.py", which, cwd=project, env=env).stdout.strip().splitlines()[-1]


@pytest.mark.timeout(240)
def test_a_spectrum_like_argument_keys_hits_and_misses_on_change(tmp_path):
    pytest.importorskip("numpy")
    project, env = _setup(tmp_path)
    # An equal spectrum built again is the same call; the result is plain Python's.
    assert _run(project, env, "same") == "[(9.0, 9.0), (9.0, 9.0)] 1 1 []"
    # Keyed alike in a new process, where every class and lambda is new.
    assert _run(project, env, "same") == "[(9.0, 9.0), (9.0, 9.0)] 2 0 []"
    # The flux, the axis values and what the class was built with all key.
    assert _run(project, env, "flux") == "[(10.0, 10.0)] 0 1 []"
    assert _run(project, env, "axis") == "[(11.0, 11.0)] 0 1 []"
    assert _run(project, env, "unit") == "[(9.0, 9.0)] 0 1 []"
    assert _run(project, env, "unit") == "[(9.0, 9.0)] 1 0 []"


@pytest.mark.timeout(240)
def test_what_such_a_class_holds_that_cannot_be_keyed_still_refuses_the_key(tmp_path):
    pytest.importorskip("numpy")
    project, env = _setup(tmp_path)
    assert _run(project, env, "lock") == "[(9.0, 9.0)] 0 1 ['KEY-UNHASHABLE-ARG']"


def test_the_users_own_unnamed_class_is_not_keyed_this_way():
    from cash.unnamed_code import library_code_override

    override = library_code_override()

    def make():
        return type("Local", (), {"k": 1})

    # Defined in this test module, which is the user's: pickled as pickle does.
    assert override(make()) is NotImplemented
    assert override(lambda x: x) is NotImplemented
