"""A file that was looked for and was not there is an input like any other.

Round-16 gate finding (WRONG ANSWER, silent, 4/4). A cached function reading a
per-directory config by RELATIVE name -- the ordinary "does this directory have
a config?" shape -- run in directory A, then B (which has no such file), then A
again, was served **B's answer in A**, as a cache hit, with no warning::

    run 1  cwd=dirA (cfg.txt = 7)   ran, 7 * base          correct
    run 2  cwd=dirA                 HIT, 7 * base          correct (control)
    run 3  cwd=dirB (no cfg.txt)    ran, 1 * base          correct
    run 4  cwd=dirA                 HIT, 1 * base          *** WRONG ***

The mechanism, from the stored metadata rather than from reasoning: the dirA
entry recorded ``{abs/cfg.txt: {...}, 'cfg.txt': {...}}``. The dirB run
recorded ``deps=None`` -- it read nothing, so there was nothing to track -- and
that entry, carrying no dependencies at all, is valid everywhere for ever.

Absence was the one input cash could not see. It is recorded now
(``{'absent': True}``), it re-resolves against the live cwd like any other
relative dep, and it is stale the moment the file appears.

The reverse order matters as much: first run in B (absent), then A (present).
The entry from B is invalidated by A's file existing, so that direction is
covered by the same record -- which is why this is done on the dependency side
rather than by folding the cwd into the key, where the first-run-in-B ordering
would still have been wrong.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from cash import Cash


@pytest.fixture
def cash_instance(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _optional_config_reader(c, runs, name="cfg.txt"):
    @c.cache(assume_safe=True)
    def scaled(n):
        runs.append(n)
        if os.path.exists(name):
            with open(name) as fh:
                return n * int(fh.read().strip())
        return n

    return scaled


def test_a_file_that_appears_invalidates_the_entry(cash_instance, tmp_path, monkeypatch):
    """The B -> A direction, in process: computed without the file, then with."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runs: list[int] = []
    scaled = _optional_config_reader(cash_instance, runs)

    assert scaled(1000) == 1000            # no cfg.txt: the defaults branch
    (work / "cfg.txt").write_text("7", encoding="utf-8")

    assert scaled(1000) == 7000, "the entry computed without the file was served"
    assert len(runs) == 2


def test_a_file_that_disappears_invalidates_too(cash_instance, tmp_path, monkeypatch):
    """The other direction, which already worked -- kept so it cannot regress."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "cfg.txt").write_text("7", encoding="utf-8")
    monkeypatch.chdir(work)
    runs: list[int] = []
    scaled = _optional_config_reader(cash_instance, runs)

    assert scaled(1000) == 7000
    (work / "cfg.txt").unlink()

    assert scaled(1000) == 1000
    assert len(runs) == 2


def test_nothing_changing_still_hits(cash_instance, tmp_path, monkeypatch):
    """The control. Every assertion above passes if nothing caches at all."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runs: list[int] = []
    scaled = _optional_config_reader(cash_instance, runs)

    assert scaled(1000) == 1000
    assert scaled(1000) == 1000

    assert len(runs) == 1, f"the second call recomputed: {runs}"


def test_an_unrelated_file_appearing_does_not_churn(cash_instance, tmp_path, monkeypatch):
    """Only what the call looked for counts, not everything in the directory."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runs: list[int] = []
    scaled = _optional_config_reader(cash_instance, runs)

    assert scaled(1000) == 1000
    (work / "something_else.txt").write_text("hello", encoding="utf-8")
    assert scaled(1000) == 1000

    assert len(runs) == 1, f"an unrelated file invalidated the entry: {runs}"


def test_isfile_is_covered_too(cash_instance, tmp_path, monkeypatch):
    """`os.path.isfile` is the same idiom and must record the same thing."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runs: list[int] = []

    @cash_instance.cache(assume_safe=True)
    def scaled(n):
        runs.append(n)
        return n * 3 if os.path.isfile("flag.txt") else n

    assert scaled(10) == 10
    (work / "flag.txt").write_text("x", encoding="utf-8")
    assert scaled(10) == 30
    assert len(runs) == 2


def test_a_present_file_is_recorded_as_read_not_as_absent(cash_instance, tmp_path, monkeypatch):
    """A probe that succeeds is followed by the read, which is the real record.

    Without this, the absent-marker could shadow a genuine content dependency
    and an EDIT to the file would stop invalidating -- the opposite failure.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "cfg.txt").write_text("7", encoding="utf-8")
    monkeypatch.chdir(work)
    runs: list[int] = []
    scaled = _optional_config_reader(cash_instance, runs)

    assert scaled(1000) == 7000
    (work / "cfg.txt").write_text("9", encoding="utf-8")
    assert scaled(1000) == 9000, "editing the file no longer invalidates"
    assert len(runs) == 2


# --------------------------------------------------------------------------- #
# The reported sequence, in real processes                                     #
# --------------------------------------------------------------------------- #

# The sleep is load-bearing across processes: cash promotes a result past RAM
# only when computing it cost more than restoring it will, so a multiplication
# is never written to disk and every arm below would "recompute" for a reason
# that has nothing to do with the dependency under test. The tester's own first
# probe was wrong this way; the control run asserts the HIT that proves it.
_CHILD = """
import os, sys, time
import cash

@cash.cache(assume_safe=True)
def scaled(n):
    print("RAN", file=sys.stderr, flush=True)
    time.sleep(0.3)
    if os.path.exists("cfg.txt"):
        with open("cfg.txt") as fh:
            return n * int(fh.read().strip())
    return n

print("RESULT", scaled(1000))
"""


def test_the_reported_sequence_across_processes(tmp_path):
    """A -> A -> B -> A, each a fresh interpreter, one cache directory.

    The in-process arms above cannot show this: the reported bug is about what
    a SECOND process reads back out of the entry the first one wrote.
    """
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(_CHILD), encoding="utf-8")
    dir_a = tmp_path / "dirA"
    dir_b = tmp_path / "dirB"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "cfg.txt").write_text("7", encoding="utf-8")
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / "cache"))

    def run(cwd):
        proc = subprocess.run([sys.executable, str(script)], cwd=str(cwd), env=env,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc.stdout.strip(), proc.stderr.count("RAN")

    assert run(dir_a) == ("RESULT 7000", 1)
    assert run(dir_a) == ("RESULT 7000", 0), "control: the second run must HIT"
    assert run(dir_b) == ("RESULT 1000", 1)
    assert run(dir_a) == ("RESULT 7000", 1), (
        "directory B's answer was served in directory A"
    )
