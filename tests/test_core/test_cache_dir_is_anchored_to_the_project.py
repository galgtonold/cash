"""Where the cache lives is a property of the CODE, not of your cwd.

Round-14 and round-15 gate findings (CAS-84, CAS-99), one problem in two
halves.

The default cache directory used to be resolved from ``os.getcwd()``, so
running the same script from somewhere else quietly started a second cache:
``6 of 6 restored`` became ``0 of 6``, a fresh 232MB ``.cash`` appeared where
the job happened to be standing, nothing warned, and it was indistinguishable
from an ordinary cold run. Three separate round-15 projects hit it; one wrote a
``.cash`` at the drive root. A cron job, a CI step and a colleague's terminal
are all "somewhere else".

The documented escape hatch -- ``[tool.cash] cache_dir`` in ``pyproject.toml``
-- was discovered by walking up from the cwd too, so it was ignored in exactly
the case that needed it: a user who did everything right (absolute script path,
absolute ``cache_dir``, config committed to the repo) still got a cold cache.

Subprocesses throughout. This is a property of a fresh interpreter deciding
where its cache lives; an in-process test asserting on ``get_config()`` cannot
see what a real second run does, and that is what the reporter measured.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# A tiny project: a script that caches one call and prints whether it ran.
#
# The sleep is load-bearing: cash promotes a result past RAM only when the
# compute was expensive enough to be worth the disk I/O (a 0.1s floor), and a
# doubling is not. Without it every arm below recomputes for a reason that has
# nothing to do with WHERE the cache is -- which is what the first draft of
# this file measured.
_SCRIPT = """
import sys, time
import cash

@cash.cache(assume_safe=True)
def work(n):
    print("RAN", file=sys.stderr, flush=True)
    time.sleep(0.3)
    return n * 2

print("RESULT", work(21), flush=True)
print("CACHE_DIR", cash.get_config().cache_dir, flush=True)
"""


@pytest.fixture
def project(tmp_path):
    """An ordinary project: a pyproject.toml and a script one level down."""
    root = tmp_path / "proj"
    (root / "scripts").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n', encoding="utf-8",
    )
    script = root / "scripts" / "job.py"
    script.write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return root, script, elsewhere


def _run(script, cwd, env_extra=None):
    env = dict(os.environ)
    env.pop("CASH_CACHE_DIR", None)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=str(cwd), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    lines = dict(
        line.split(" ", 1) for line in proc.stdout.splitlines() if " " in line
    )
    return {
        "ran": "RAN" in proc.stderr,
        "cache_dir": lines.get("CACHE_DIR", ""),
        "result": lines.get("RESULT", ""),
        "stderr": proc.stderr,
    }


def test_the_same_script_uses_one_cache_from_any_directory(project):
    """The reported failure: three cwds, three caches, every run cold."""
    root, script, elsewhere = project

    first = _run(script, cwd=root)
    second = _run(script, cwd=elsewhere)
    third = _run(script, cwd=root / "scripts")

    assert first["ran"], "the first run must actually compute something"
    assert not second["ran"], (
        "running from another directory recomputed -- it found a different cache"
    )
    assert not third["ran"]
    assert first["result"] == second["result"] == third["result"] == "42"

    dirs = {r["cache_dir"] for r in (first, second, third)}
    assert len(dirs) == 1, f"one project, {len(dirs)} cache directories: {dirs}"
    assert os.path.dirname(dirs.pop()) == str(root), (
        "the cache belongs beside the project, not beside the caller"
    )


def test_no_stray_cache_appears_where_the_job_was_launched(project):
    """The other half of the symptom: duplicate caches filling the disk."""
    root, script, elsewhere = project
    _run(script, cwd=elsewhere)
    assert not (elsewhere / ".cash").exists(), (
        f"a second cache was created in the launch directory: "
        f"{list(elsewhere.iterdir())}"
    )


def test_a_relative_cache_dir_in_pyproject_is_relative_to_pyproject(project):
    """CAS-99: the documented fix, working from outside the project."""
    root, script, elsewhere = project
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n'
        '[tool.cash]\ncache_dir = "var/cache"\n', encoding="utf-8",
    )

    inside = _run(script, cwd=root)
    outside = _run(script, cwd=elsewhere)

    assert inside["cache_dir"] == outside["cache_dir"], (
        "the project's own config resolved to two different directories"
    )
    assert inside["cache_dir"] == str(root / "var" / "cache")
    assert not outside["ran"], "the second run did not find the first one's cache"


def test_an_absolute_cache_dir_in_pyproject_is_honoured_from_outside(project, tmp_path):
    """The reporter's exact case: absolute path, committed config, wrong cwd."""
    root, script, elsewhere = project
    shared = tmp_path / "shared-cache"
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n'
        # A TOML LITERAL string (single quotes): Python's repr would escape the
        # Windows backslashes and TOML would hand back the doubled form.
        f"[tool.cash]\ncache_dir = '{shared}'\n", encoding="utf-8",
    )

    first = _run(script, cwd=root)
    second = _run(script, cwd=elsewhere)

    assert first["cache_dir"] == second["cache_dir"] == str(shared)
    assert first["ran"] and not second["ran"]


def test_the_env_var_still_wins_and_stays_relative_to_you(project, tmp_path):
    """``CASH_CACHE_DIR`` is typed in the shell you are in, so it means there.

    The control for the anchoring: it must not start rewriting paths the user
    gave explicitly, which would be the same surprise in the other direction.
    """
    root, script, elsewhere = project
    result = _run(script, cwd=elsewhere, env_extra={"CASH_CACHE_DIR": "here"})
    assert result["cache_dir"] == "here", (
        "an explicitly given path must be carried exactly as written"
    )
    # And what it resolves to on disk is the caller's directory, not the
    # project's -- the assertion that would catch the anchoring reaching a
    # value the user typed.
    assert (elsewhere / "here").is_dir()
    assert not (root / "here").exists()


def test_a_script_without_a_project_anchors_beside_itself(tmp_path):
    """No pyproject, no .git -- the script's own directory is still stable."""
    loose = tmp_path / "tools"
    loose.mkdir()
    script = loose / "solo.py"
    script.write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    elsewhere = tmp_path / "away"
    elsewhere.mkdir()

    first = _run(script, cwd=loose)
    second = _run(script, cwd=elsewhere)

    assert first["cache_dir"] == second["cache_dir"] == str(loose / ".cash")
    assert not second["ran"]


def test_an_interactive_session_still_uses_the_cwd(tmp_path):
    """``python -c`` has no ``__main__.__file__``: nothing to anchor to.

    This is also the notebook's path -- the kernel launcher lives inside the
    interpreter's own installation, so a notebook keeps caching next to itself
    exactly as before. If this regressed, every notebook user's cache would
    move on upgrade.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         "import cash; print(cash.get_config().cache_dir)"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300,
        env={k: v for k, v in os.environ.items() if k != "CASH_CACHE_DIR"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path / ".cash")


# --------------------------------------------------------------------------- #
# __main__.__file__ is not always a script                                    #
# --------------------------------------------------------------------------- #

def test_an_ipython_startup_script_does_not_become_the_anchor(tmp_path, monkeypatch):
    """IPython SETS ``__main__.__file__`` while running its startup scripts.

    Found by the notebook integration suite, not by reasoning: with the anchor
    reading ``__main__.__file__`` unconditionally, a kernel that imported cash
    from a profile startup file resolved its cache to
    ``~/.ipython/profile_default/startup/.cash`` and every entry went there.
    The symptom was a notebook that stopped hitting after a kernel restart --
    two directories, no error.

    An interactive session has no running script by definition, so the answer
    is the cwd, which is what a notebook has always used.
    """
    import types

    from cash import config as cash_config

    startup = tmp_path / ".ipython" / "profile_default" / "startup"
    startup.mkdir(parents=True)
    (startup / "00-cash.py").write_text("import cash\n", encoding="utf-8")
    main = types.ModuleType("__main__")
    main.__file__ = str(startup / "00-cash.py")
    monkeypatch.setitem(sys.modules, "__main__", main)

    fake_ipython = types.ModuleType("IPython")
    fake_ipython.get_ipython = lambda: object()          # a live shell
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    assert cash_config.project_anchor() == work.resolve()


def test_a_real_script_is_still_the_anchor_without_a_shell(tmp_path, monkeypatch):
    """The control: with no interactive shell, ``__main__`` is a script again.

    Without this, the test above passes on an anchor that has stopped looking
    at the running script at all.
    """
    import types

    from cash import config as cash_config

    project = tmp_path / "proj"
    (project / "scripts").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    main = types.ModuleType("__main__")
    main.__file__ = str(project / "scripts" / "job.py")
    monkeypatch.setitem(sys.modules, "__main__", main)

    fake_ipython = types.ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None              # no shell
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.chdir(tmp_path)

    assert cash_config.project_anchor() == project.resolve()

# --------------------------------------------------------------------------- #
# Spawned workers must anchor where their parent did                           #
# --------------------------------------------------------------------------- #

_POOL_MAIN = """
from concurrent.futures import ProcessPoolExecutor

from .work import where


def main():
    print("PARENT", where(), flush=True)
    with ProcessPoolExecutor(max_workers=1) as pool:
        print("WORKER", list(pool.map(where, [1]))[0], flush=True)


if __name__ == "__main__":
    main()
"""

_POOL_WORKER = """
import cash


def where(_=None):
    return cash.get_config().cache_dir
"""


def test_a_spawned_worker_anchors_where_its_parent_did(tmp_path):
    """One run, one cache directory -- even across a process pool.

    Run as ``python -m pkg``, a spawned worker has no ``__main__.__file__`` and
    no ``__spec__``, so the anchor fell back to the cwd while the parent had
    anchored to its project: one fan-out wrote into two cache directories and
    neither side could see the other's entries. A round-16 tester measured that
    3/3 and named the cost -- the whole point of a shared cache across workers,
    defeated silently.

    The ``-m`` form is load-bearing here. With a plain ``python run.py`` parent
    the child DOES inherit ``__main__.__file__`` and the two already agreed, so
    a fixture built that way passes against the unfixed code. What the worker
    always inherits is ``sys.argv[0]``, which is what the anchor falls back to
    now.
    """
    project = tmp_path / "proj"
    pkg = project / "pkg"
    pkg.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n', encoding="utf-8",
    )
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__main__.py").write_text(textwrap.dedent(_POOL_MAIN), encoding="utf-8")
    (pkg / "work.py").write_text(textwrap.dedent(_POOL_WORKER), encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = dict(os.environ, PYTHONPATH=str(project))
    env.pop("CASH_CACHE_DIR", None)

    proc = subprocess.run([sys.executable, "-m", "pkg"], cwd=str(elsewhere),
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]

    lines = dict(line.split(" ", 1) for line in proc.stdout.splitlines() if " " in line)
    assert lines["PARENT"] == lines["WORKER"], (
        f"one run resolved two cache directories: {lines}"
    )
    assert lines["PARENT"] == str(project / ".cash")


def test_a_lint_only_pyproject_below_the_project_is_not_a_project(tmp_path):
    """Round 19: `tests/pyproject.toml` holding only `[tool.ruff]` made
    `tests/` the project whenever a script there ran -- a second, cold cache,
    and the repository's `[tool.cash]` ignored. A pyproject.toml marks a
    project when it has `[project]`, `[build-system]`, `[tool.poetry]` or
    `[tool.cash]`."""
    root = tmp_path / "repo"
    tests = root / "tests"
    tests.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "r"\nversion = "0"\n\n[tool.cash]\ncache_dir = "shared_cache"\n',
        encoding="utf-8")
    (tests / "pyproject.toml").write_text('[tool.ruff]\nline-length = 100\n', encoding="utf-8")
    (tests / "job.py").write_text(_SCRIPT, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    p = subprocess.run([sys.executable, "job.py"], cwd=str(tests), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    cache_dir = next(line.split(" ", 1)[1] for line in p.stdout.splitlines()
                     if line.startswith("CACHE_DIR"))
    assert os.path.normcase(os.path.realpath(cache_dir)) == \
        os.path.normcase(os.path.realpath(root / "shared_cache")), cache_dir
