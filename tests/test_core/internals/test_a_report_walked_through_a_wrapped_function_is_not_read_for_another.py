"""A stored purity report walked through the function a callable object
wraps is not read back once that object wraps another function.

The walk follows what a global the body calls runs (``VEC =
np.vectorize(double)``: ``double``), but the report's guard held that
global by its type alone, and ``np.vectorize(triple)`` has the same type.
So a later process bound to ``triple`` read the report walked through
``double``: its helpers named ``double`` and ``triple`` was never audited
(the key itself still moved, with the global's own digest). Whether the
first walk was stored hung on its CPU time (`report_store._READ_COST`), so
the miss's "it now uses helper triple" came and went between CI runs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core]

_JOB = """
    import sys

    import cash
    from cash.analysis import report_store

    from fns_double import double
    from fns_triple import triple

    # Store every walk, however cheap: whether a walk is stored is not
    # what this test is about, and a fast machine leaves this one out.
    report_store._READ_COST = -1.0

    {make}


    @cash.cache
    def scaled(n):
        return float(sum(VEC([0, 1, 2, n])))


    value = scaled(4)
    report_store.flush()
    stats = report_store.status()
    print(value, scaled.cache_info()["changed"], "stale" if stats["stale"] else "fresh", stats["loaded"] > 0)
"""

_VECTORIZE = """
    import numpy as np

    VEC = np.vectorize({"double": double, "triple": triple}[sys.argv[1]], otypes=[float])
"""

# A class-based wrapper of the user's: the function it holds is a layer too.
_OWN_WRAPPER = """
    class Each:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, xs):
            return [self.fn(x) for x in xs]


    VEC = Each({"double": double, "triple": triple}[sys.argv[1]])
"""


def _project(folder: Path, make: str) -> Path:
    folder.mkdir()
    (folder / "fns_double.py").write_text("def double(x):\n    return 2 * x\n", encoding="utf-8")
    (folder / "fns_triple.py").write_text("def triple(x):\n    return 3 * x\n", encoding="utf-8")
    job = textwrap.dedent(_JOB).replace("{make}", textwrap.dedent(make).strip())
    (folder / "job.py").write_text(job, encoding="utf-8")
    return folder


def _run(folder: Path, which: str) -> str:
    return run_python("job.py", which, cwd=folder).stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("make", [_VECTORIZE, _OWN_WRAPPER], ids=["np.vectorize", "own wrapper"])
def test_the_report_of_a_global_wrapping_another_function_is_walked_again(tmp_path, make):
    if make is _VECTORIZE:
        pytest.importorskip("numpy")
    folder = _project(tmp_path / "p", make)
    assert _run(folder, "double") == "14.0 {} fresh False"
    # Bound to triple: the report walked through double is stale, and the
    # miss names both helpers.
    assert _run(folder, "triple").startswith(
        "21.0 {'global VEC changed; it no longer uses helper fns_double.double; "
        "it now uses helper fns_triple.triple': 1} stale"
    )
    # Each binding reads its own report back, and hits.
    assert _run(folder, "triple") == "21.0 {} fresh True"
    assert _run(folder, "double") == "14.0 {} fresh True"
