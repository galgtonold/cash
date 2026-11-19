"""A function the user's code made in a method, held by an argument, keys.

lmfit's ``CompositeModel`` keeps a function it defines in its ``__init__``.
In a walk of lmfit's own repository lmfit is the user's code, and pickle
finds no such function by name, so every call handed a composite model ran
uncached (KEY-UNHASHABLE-ARG). Inside a cached call's arguments such a
function is keyed by what defines it; the code-argument fold keys its code
and the globals that code reads. A change to either, or a closure holding
other values, keys apart; a closure holding what cannot be keyed still
refuses the key.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core]

PACKAGE = """
    import threading

    FACTOR = {factor}


    def make(k):
        def scale(x):
            return x * k * FACTOR

        return scale


    class Model:
        def __init__(self, a, b, lock=False):
            guard = threading.Lock() if lock else None

            def _shift(x):
                return x + {shift} if guard is None or guard else x

            self.shift = _shift
            self.a = make(a)
            self.b = make(b)

        def run(self, x):
            return self.a(self.shift(x)) - self.b(x)
"""

MAIN = """
    import sys
    import warnings

    import cash
    from pkg import Model

    warnings.simplefilter("always")


    @cash.cache
    def fit(model, x):
        return model.run(x)


    a, b, lock = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "lock"
    with warnings.catch_warnings(record=True) as caught:
        out = [fit(Model(a, b, lock), 2) for _ in range(2)]
    info = fit.cache_info()
    codes = sorted({str(getattr(w.message, "code", "")) for w in caught} - {""})
    print(out, info["hits"], info["misses"], codes)
"""


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "pkg").mkdir(parents=True)
    (project / "main.py").write_text(textwrap.dedent(MAIN), encoding="utf-8")
    _write_package(project, factor=4, shift=5)
    return project


def _write_package(project: Path, factor: int, shift: int) -> None:
    source = textwrap.dedent(PACKAGE).format(factor=factor, shift=shift)
    (project / "pkg" / "__init__.py").write_text(source, encoding="utf-8")


def _run(project: Path, a: int, b: int, lock: str = "-") -> str:
    return run_python("main.py", str(a), str(b), lock, cwd=project).stdout.strip().splitlines()[-1]


@pytest.mark.timeout(240)
def test_it_hits_and_misses_on_what_defines_it(tmp_path):
    project = _project(tmp_path)
    # (2 + 5) * 1 * 4 - 2 * 2 * 4: the second call of a process hits.
    assert _run(project, 1, 2) == "[12, 12] 1 1 []"
    # Every function is new in a new process; the key is not.
    assert _run(project, 1, 2) == "[12, 12] 2 0 []"
    # The closures swapped hold other values.
    assert _run(project, 2, 1) == "[48, 48] 1 1 []"
    # A global the made function's code reads.
    _write_package(project, factor=5, shift=5)
    assert _run(project, 1, 2) == "[15, 15] 1 1 []"
    # The made function's own body.
    _write_package(project, factor=5, shift=6)
    assert _run(project, 1, 2) == "[20, 20] 1 1 []"


@pytest.mark.timeout(240)
def test_a_capture_that_cannot_be_keyed_still_refuses_the_key(tmp_path):
    project = _project(tmp_path)
    assert _run(project, 1, 2, "lock") == "[12, 12] 0 2 ['KEY-UNHASHABLE-ARG']"
