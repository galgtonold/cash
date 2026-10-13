"""A pool worker caches in the parent's cache directory, however it was started.

A joblib (loky) worker is launched as ``python -m <joblib's launcher>``, so
its ``__main__`` is installed code. Resolved like any process, it was "the
installed tool joblib": with no project above the script it cached in the
per-user ``~/.cache/cash/joblib`` while its parent cached beside the script,
and from inside another project it took the cwd's project. Its results were
never hits for the parent or for the next run.

A process now hands its workers the anchor it resolved (through the
environment, read only by its own multiprocessing or joblib workers), and a
worker whose launcher is installed code reads the script from ``sys.argv``.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from cash import _location
from tests._scripts import run_python

pytest.importorskip("joblib")

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

_LIB = """
import sys
import cash

@cash.cache(assume_safe=True)
def work(x):
    print("RUN", x, file=sys.stderr)
    return x * x
"""

_SCRIPT = """
import sys
import multiprocessing
import concurrent.futures
from joblib import Parallel, delayed
import lib

if __name__ == "__main__":
    print(Parallel(n_jobs=2)(delayed(lib.work)(i) for i in (1, 2)))
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        print(pool.map(lib.work, (3, 4)))
    if sys.platform != "win32":
        ctx = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(2, mp_context=ctx) as ex:
            print(list(ex.map(lib.work, (5, 6))))
    print("PARENT", file=sys.stderr)
    print([lib.work(i) for i in range(1, 7 if sys.platform != "win32" else 5)])
"""


def _no_project_above(path: Path) -> bool:
    return not any((d / m).exists() for d in [path, *path.parents] for m in _location.PROJECT_MARKERS)


def _script_dir(tmp_path: Path) -> Path:
    where = tmp_path / "scratch"
    where.mkdir()
    (where / "lib.py").write_text(textwrap.dedent(_LIB), encoding="utf-8")
    (where / "run.py").write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    if not _no_project_above(where):
        pytest.skip("the temp dir is inside a project")
    return where


def _run(script_dir: Path, cwd: Path, tmp_path: Path):
    # No CASH_CACHE_DIR: the default location is what is under test. The
    # per-user cache root goes to the test's own folder.
    env = {"XDG_CACHE_HOME": str(tmp_path / "user-cache"), "LOCALAPPDATA": str(tmp_path / "user-cache")}
    return run_python(script_dir / "run.py", cwd=cwd, cache_dir=None, env=env, timeout=240)


def _assert_shared(run, script_dir: Path, tmp_path: Path) -> None:
    assert "RUN" not in run.stderr.split("PARENT", 1)[1], "the parent recomputed what a worker stored:\n" + run.stderr
    assert (script_dir / ".cash").is_dir()
    if sys.platform.startswith("linux"):
        assert not (tmp_path / "user-cache").exists(), "a worker cached in the per-user folder"


def test_workers_of_a_script_with_no_project_cache_beside_it(tmp_path):
    where = _script_dir(tmp_path)
    run = _run(where, where, tmp_path)
    _assert_shared(run, where, tmp_path)
    again = _run(where, where, tmp_path)
    assert "RUN" not in again.stderr, "the next run recomputed:\n" + again.stderr


def test_workers_ignore_the_project_the_cwd_is_in(tmp_path):
    where = _script_dir(tmp_path)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "pyproject.toml").write_text('[project]\nname = "other"\n', encoding="utf-8")
    run = _run(where, elsewhere, tmp_path)
    _assert_shared(run, where, tmp_path)
    assert not (elsewhere / ".cash").exists(), "a worker cached in the cwd's project"


def test_workers_agree_when_the_parent_never_used_cash_first(tmp_path):
    """Workers started before the parent resolved anything read the script
    from ``sys.argv``: the parent's later calls still hit."""
    where = _script_dir(tmp_path)
    (where / "run.py").write_text(
        textwrap.dedent("""
        import sys
        from joblib import Parallel, delayed

        def via_lib(x):
            import lib
            return lib.work(x)

        if __name__ == "__main__":
            print(Parallel(n_jobs=2)(delayed(via_lib)(i) for i in (1, 2, 3)))
            print("PARENT", file=sys.stderr)
            import lib
            print([lib.work(i) for i in (1, 2, 3)])
        """),
        encoding="utf-8",
    )
    run = _run(where, where, tmp_path)
    _assert_shared(run, where, tmp_path)


def test_a_subprocess_of_a_worker_is_not_taken_for_one(monkeypatch, tmp_path):
    """Only the parent's own workers read what it handed on: a process that is
    no multiprocessing worker, or the worker of another, resolves as itself."""
    monkeypatch.setenv(_location.LAUNCH_ENV, '{"pid": 1, "anchor": "/nowhere", "entry": null}')
    monkeypatch.setattr(_location, "_PARENT_LAUNCH", None)
    assert _location.project_anchor() != Path("/nowhere")
