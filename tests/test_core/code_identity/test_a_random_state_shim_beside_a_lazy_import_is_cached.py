"""A call through a random-state shim with a lazy import in its tree is cached.

Reported against scanpy checked out as the user's code: its legacy
``random_state`` shim is a numpy ``Generator`` subclass whose methods, made
in a loop, forward each draw with ``getattr(self.state, name)(...)``. A call
whose helpers reach it is served on a trace of what it ran. ``pca`` also
holds ``from ._dask import ...`` in a branch for dask arrays, a module not
loaded yet whose top level cash cannot read: the two together refused every
call (KEY-UNTRACKABLE-DEP), though either alone is left to the trace or
named as the import.

On Python 3.12 and newer the lookups trace the call anyway, and an import
that runs is in that trace: a seeded call hits in the next process and an
edit to the lazily imported module recomputes. An unseeded draw is frozen
with a warning, as everywhere else.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

TRACED = sys.version_info >= (3, 12)

RNG = textwrap.dedent("""
    import numpy as np


    class LegacyRng(np.random.Generator):
        def __init__(self, seed):
            self.seed = seed
            self.state = np.random.mtrand._rand if seed is None else np.random.RandomState(seed)

        @classmethod
        def _delegate(cls):
            for name, meth in np.random.Generator.__dict__.items():
                if name.startswith("_") or not callable(meth) or name in cls.__dict__:
                    continue

                def make(name):
                    def wrapper(self, *args, **kwargs):
                        return getattr(self.state, name)(*args, **kwargs)

                    return wrapper

                setattr(cls, name, make(name))


    LegacyRng._delegate()
""")

# Not readable without running it: a call cash does not know at the top level.
SPARSE = textwrap.dedent("""
    import json

    FACTOR = json.loads("{FACTOR}")


    def scale(values):
        return [v * FACTOR for v in values]
""")

OPS = textwrap.dedent("""
    from .rng import LegacyRng


    def project(values, seed, sparse):
        rng = LegacyRng(seed)
        noise = rng.normal(size=len(values))
        out = [round(v + n, 6) for v, n in zip(values, noise)]
        if sparse:
            from ._sparse import scale

            out = scale(out)
        return out
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from rngpkg.ops import project

    @cash.cache
    def embed(values, seed=0, sparse=False):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return project(values, seed, sparse)

    seed = None if sys.argv[1] == "none" else int(sys.argv[1])
    print(embed([1.0, 2.0], seed=seed, sparse=sys.argv[2] == "sparse"))
""")


def _write(tmp_path, *, factor=2):
    pkg = tmp_path / "rngpkg"
    pkg.mkdir(exist_ok=True)
    files = {
        "__init__.py": "",
        "rng.py": RNG,
        "_sparse.py": SPARSE.replace("{FACTOR}", str(factor)),
        "ops.py": OPS,
    }
    for name, text in files.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")


def _run(tmp_path, seed="0", sparse="dense", *, cash=True):
    if cash:
        proc = run_python("main.py", seed, sparse, cwd=tmp_path, env={"CASH_SUMMARY": "1"})
    else:
        source = (tmp_path / "main.py").read_text(encoding="utf-8")
        plain = source.replace("import cash\n", "").replace("@cash.cache\n", "")
        proc = run_python("-c", plain, seed, sparse, cwd=tmp_path)
    return proc.stdout.strip().splitlines()[0], proc.stderr


@pytest.mark.skipif(not TRACED, reason="a trace of what a call ran needs sys.monitoring (Python 3.12+)")
@pytest.mark.parametrize("sparse", ["dense", "sparse"])
def test_a_seeded_call_hits_in_the_next_process_with_the_plain_result(tmp_path, sparse):
    _write(tmp_path)
    plain, _ = _run(tmp_path, sparse=sparse, cash=False)
    out, err = _run(tmp_path, sparse=sparse)
    assert out == plain and "[RUN]" in err
    out, err = _run(tmp_path, sparse=sparse)
    assert out == plain
    assert "[RUN]" not in err, f"an unedited seeded run must hit:\n{err}"


@pytest.mark.skipif(not TRACED, reason="a trace of what a call ran needs sys.monitoring (Python 3.12+)")
def test_an_edit_to_the_lazily_imported_module_recomputes(tmp_path):
    _write(tmp_path)
    first, _ = _run(tmp_path, sparse="sparse")
    _run(tmp_path, sparse="sparse")
    _write(tmp_path, factor=10)
    want, _ = _run(tmp_path, sparse="sparse", cash=False)
    out, err = _run(tmp_path, sparse="sparse")
    assert out == want != first, f"served the result from before the edit: {out}\n{err}"
    assert "[RUN]" in err


def test_an_unseeded_draw_is_frozen_with_a_warning(tmp_path):
    """A draw from the global stream no seed stands behind keeps the first
    value, as any unseeded draw does, and says so (RANDOM-UNSEEDED). With
    no trace (before 3.12) the lookups by name run it every time."""
    _write(tmp_path)
    first, err = _run(tmp_path, seed="none")
    assert "[RUN]" in err
    if not TRACED:
        _, err = _run(tmp_path, seed="none")
        assert "[RUN]" in err and "code picked by a run-time name" in err, err
        return
    assert "RANDOM-UNSEEDED" in err, err
    out, _ = _run(tmp_path, seed="none")
    assert out == first
