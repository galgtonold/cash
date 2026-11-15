"""A lazy import in code the call never runs does not stop it being cached.

Reported against geopandas: building a ``GeoDataFrame`` and calling
``to_crs`` reaches, through methods it never calls, a function that imports
``geopandas.io._geoarrow`` lazily. That module's top level imports one of
the package's own modules whose top level calls code cash does not know, so
its source cannot vouch for what it binds, and every call ran uncached.

Until such a module is loaded none of its code can run. A call that leaves
it unloaded is stored by its key; a call that loads it is not stored, and
the next call keys the module's code, as before. On every Python, traced or
not, through the body or through an argument's methods.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

# Not readable without running it: a call cash does not know at the top level.
ARROW = textwrap.dedent("""
    import json

    SCALE = json.loads("{SCALE}")


    def to_arrow(values):
        return [SCALE * v for v in values]
""")

FRAME = textwrap.dedent("""
    class Frame:
        def __init__(self, values):
            self.values = list(values)

        def reproject(self, k):
            return Frame(v * k for v in self.values)

        def to_arrow(self):
            from geopkg._arrow import to_arrow

            return to_arrow(self.values)
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from geopkg.frame import Frame

    @cash.cache
    def build(n, arrow):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        frame = Frame(range(n)).reproject(2)
        return frame.to_arrow() if arrow else frame.values

    @cash.cache
    def through(frame, arrow):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        out = frame.reproject(3)
        return out.to_arrow() if arrow else out.values

    which = build if sys.argv[1] == "build" else lambda n, a: through(Frame(range(n)), a)
    for arrow in sys.argv[2:]:
        print(which(3, arrow == "1"))
    print("loaded" if "geopkg._arrow" in sys.modules else "unloaded")
""")


def _write(tmp_path, scale=10):
    pkg = tmp_path / "geopkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "_arrow.py").write_text(ARROW.replace("{SCALE}", str(scale)), encoding="utf-8")
    (pkg / "frame.py").write_text(FRAME, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")


def _run(tmp_path, *args, trace="1"):
    proc = run_python("main.py", *args, cwd=tmp_path, env={"CASH_TRACE": trace})
    return proc.stdout.split("\n")[:-1], proc.stderr.count("[RUN]"), proc.stderr


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("how", ["build", "through"])
def test_calls_that_never_import_it_hit(tmp_path, how, trace):
    _write(tmp_path)
    out, runs, err = _run(tmp_path, how, "0", "0", trace=trace)
    assert out[-1] == "unloaded" and len(set(out[:-1])) == 1
    assert runs == 1, err
    out, runs, err = _run(tmp_path, how, "0", trace=trace)
    assert runs == 0, f"a second process must hit:\n{err}"


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("how", ["build", "through"])
def test_the_call_that_imports_it_is_not_stored(tmp_path, how, trace):
    _write(tmp_path)
    out, runs, err = _run(tmp_path, how, "1", "1", "1", trace=trace)
    assert out[-1] == "loaded"
    assert runs == 2, err
    _write(tmp_path, scale=7)
    out, runs, err = _run(tmp_path, how, "1", trace=trace)
    want = "[0, 14, 28]" if how == "build" else "[0, 21, 42]"
    assert out[0] == want, f"served a result from before the edit:\n{err}"
