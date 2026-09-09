"""``cash inspect`` / ``cash clear`` act on the cache the library wrote.

They assumed ``./.cash`` while ``cash info`` read the merged config, so the two
commands whose whole job is to act on the cache acted on a different one from
the library. A round-16 tester reproduced both halves (CAS-83):

* ``CASH_CACHE_DIR=/tmp/mycache`` then ``cash inspect`` -> "no cache found",
  while ``cash info`` printed ``/tmp/mycache`` correctly. ``cash clear --all``
  reported success having deleted a directory the user did not mean.
* ``cash clear --all <path>`` accepted both and dropped the path, clearing the
  current directory instead of the one named.

0.10.0 made this worse rather than better: the default cache now follows the
project anchor, so ``./.cash`` is the right answer less often than it was.

The subprocess is the point -- the bug lives in how the CLI resolves its
target, and calling the command function directly with a hand-built Namespace
would skip the environment and the argument parser, which is where both halves
of it live.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.core


def _run(*argv, env=None, cwd=None):
    environ = dict(os.environ)
    environ.pop("CASH_CACHE_DIR", None)
    environ.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "cash", *argv],
        capture_output=True, text=True, env=environ, cwd=cwd,
    )


@pytest.fixture
def a_cache_somewhere_else(tmp_path):
    """A populated cache at a path no cwd would guess, plus an empty cwd."""
    elsewhere = tmp_path / "elsewhere" / "cache"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    script = tmp_path / "build.py"
    script.write_text(
        "import cash, time\n"
        "@cash.cache(assume_safe=True)\n"
        "def slow(n):\n"
        "    time.sleep(0.3)\n"
        "    return n * 2\n"
        "print(slow(21))\n",
        encoding="utf-8",
    )
    env = {"CASH_CACHE_DIR": str(elsewhere)}
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, cwd=str(workdir),
                         env={**os.environ, **env})
    assert out.returncode == 0, out.stderr
    for _ in range(20):                       # the writer is a background thread
        if elsewhere.is_dir() and any(elsewhere.iterdir()):
            break
        time.sleep(0.25)
    assert any(elsewhere.iterdir()), "test setup: nothing was cached"
    return elsewhere, workdir, env


def test_inspect_reads_the_configured_cache(a_cache_somewhere_else):
    """THE BUG: inspect reported on ./.cash and found nothing."""
    elsewhere, workdir, env = a_cache_somewhere_else

    result = _run("inspect", env=env, cwd=str(workdir))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "slow" in result.stdout, (
        f"inspect did not report the entries the library wrote:\n{result.stdout}"
    )
    assert str(elsewhere) in result.stdout, (
        "inspect should name the directory it read, now that it is not the cwd"
    )


def test_clear_all_clears_the_configured_cache(a_cache_somewhere_else):
    """THE OTHER HALF: clear --all reported success having cleared nothing."""
    elsewhere, workdir, env = a_cache_somewhere_else
    decoy = workdir / ".cash"
    decoy.mkdir()
    (decoy / "CACHE_VERSION").write_text("1", encoding="utf-8")

    result = _run("clear", "--all", env=env, cwd=str(workdir))

    assert result.returncode == 0, result.stdout + result.stderr
    assert not elsewhere.exists(), "the configured cache survived clear --all"
    assert decoy.is_dir(), "clear --all removed a directory it was not pointed at"


def test_clear_all_with_a_path_is_refused(a_cache_somewhere_else):
    """Silently clearing a different cache than the one named is not an option."""
    elsewhere, workdir, env = a_cache_somewhere_else
    named = workdir / "named-cache"
    named.mkdir()
    (named / "CACHE_VERSION").write_text("1", encoding="utf-8")

    result = _run("clear", "--all", str(named), env=env, cwd=str(workdir))

    assert result.returncode == 2, result.stdout + result.stderr
    assert "mutually exclusive" in result.stdout
    assert named.is_dir(), "the named directory was cleared by a refused command"
    assert elsewhere.is_dir(), "the configured cache was cleared by a refused command"


def test_a_named_directory_still_wins(a_cache_somewhere_else):
    """The control: an explicit path is still an explicit path."""
    elsewhere, workdir, env = a_cache_somewhere_else
    named = workdir / "named-cache"
    named.mkdir()
    (named / "CACHE_VERSION").write_text("1", encoding="utf-8")

    result = _run("clear", str(named), env=env, cwd=str(workdir))

    assert result.returncode == 0, result.stdout + result.stderr
    assert not named.exists()
    assert elsewhere.is_dir(), "clearing a named path touched the configured cache"


def test_clear_all_refuses_a_directory_that_is_not_a_cache(tmp_path):
    """``--all`` deletes a path the user did not type, so it looks first.

    A mistyped ``CASH_CACHE_DIR`` used to cost nothing because the CLI ignored
    it. Now that it is honoured, a recursive delete of whatever it points at
    needs a guard.
    """
    precious = tmp_path / "not-a-cache"
    precious.mkdir()
    (precious / "thesis.txt").write_text("years of work", encoding="utf-8")

    result = _run("clear", "--all", env={"CASH_CACHE_DIR": str(precious)},
                  cwd=str(tmp_path))

    assert result.returncode == 1
    assert "does not look like a cash cache" in result.stdout
    assert (precious / "thesis.txt").exists()
