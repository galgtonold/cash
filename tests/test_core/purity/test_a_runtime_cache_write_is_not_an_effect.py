"""A write into a runtime's own cache is not the user's side effect.

numba's ``@njit(cache=True)`` writes its compiled index and data into a
``__pycache__`` next to the code (or under ``NUMBA_CACHE_DIR``), first as
``name.nbi.tmp.<hex>`` and then renamed into place. Seen by the effect
observer, every cached function calling such a kernel warned
IMPURE-OBSERVED-EFFECTS, seven lines of it, whenever numba compiled cold.
A hit that skips the write only leaves numba to compile again. Reads of
these files were already ignored; writes now are too, by the same rule.
"""

from __future__ import annotations

import importlib.util
import tempfile
import textwrap
import warnings
import zipfile

import pytest

from cash import Cash
from cash.exceptions import CashImpurityWarning
from tests._scripts import run_python


def _observed(tmp_path, target):
    """Warnings from one cached call that writes *target* through the stdlib,
    which the static pass does not walk into: only the observer can see it."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    def writes():
        with zipfile.ZipFile(target, "w") as archive:
            members = archive.namelist()
        return len(members)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        c.cache(writes)()
    assert target.exists()
    return [str(w.message) for w in record if issubclass(type(w.message), CashImpurityWarning)]


@pytest.mark.parametrize(
    "relative",
    ["__pycache__/kern.zip", "kern-3.py313.nbi", "kern-3.py313.nbc", "kern.nbi.tmp.1f2e3d", "kern.cpython-313.pyc"],
)
def test_a_write_into_a_runtime_cache_is_silent(tmp_path, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not _observed(tmp_path, target)


def test_a_write_under_numba_cache_dir_is_silent(tmp_path, monkeypatch):
    cache_dir = tmp_path / "jit-cache"
    (cache_dir / "proj").mkdir(parents=True)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache_dir))
    assert not _observed(tmp_path, cache_dir / "proj" / "index.zip")


@pytest.mark.parametrize("folder", ["__pycache__", "jit-cache"])
def test_a_nameless_temporary_file_in_a_runtime_cache_folder_is_silent(tmp_path, monkeypatch, folder):
    """numba checks the folder is writable with ``tempfile.TemporaryFile(dir=...)``,
    which on Linux opens the folder itself for writing."""
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "jit-cache"))
    where = tmp_path / folder
    where.mkdir()
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    def probes():
        with tempfile.TemporaryFile(dir=where) as fh:
            return fh.fileno() >= 0

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        c.cache(probes)()
    assert not [str(w.message) for w in record if issubclass(type(w.message), CashImpurityWarning)]


@pytest.mark.parametrize("relative", ["written.zip", "pycache_notes/kern.zip", "kern.nbi.bak", "kern.nbi.tmp.notes"])
def test_a_write_of_the_users_own_file_still_warns(tmp_path, monkeypatch, relative):
    """The control: only the runtime's caches are left out."""
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "jit-cache"))
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    warned = _observed(tmp_path, target)
    assert any("IMPURE-OBSERVED-EFFECTS" in w and target.name in w for w in warned), warned


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba is not installed")
@pytest.mark.timeout(120)
def test_a_cold_numba_cache_does_not_warn_and_the_call_caches(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "kernels.py").write_text(
        textwrap.dedent("""
        import numba

        @numba.njit(cache=True)
        def kern(a):
            s = 0.0
            for x in a:
                s += x * x
            return s
        """),
        encoding="utf-8",
    )
    (project / "main.py").write_text(
        textwrap.dedent("""
        import numpy as np
        import cash
        from kernels import kern

        @cash.cache
        def backtest(n):
            print("RAN", flush=True)
            return kern(np.arange(n, dtype=np.float64))

        print("RESULT", backtest(4), flush=True)
        """),
        encoding="utf-8",
    )
    first = run_python("main.py", cwd=project, env={"NUMBA_CACHE_DIR": None})
    assert "RAN" in first.stdout and "RESULT 14.0" in first.stdout
    assert list((project / "__pycache__").glob("*.nbi")), "numba wrote no cache: the case under test did not happen"
    assert "IMPURE-OBSERVED-EFFECTS" not in first.stderr, first.stderr
    second = run_python("main.py", cwd=project, env={"NUMBA_CACHE_DIR": None})
    assert "RAN" not in second.stdout and "RESULT 14.0" in second.stdout
