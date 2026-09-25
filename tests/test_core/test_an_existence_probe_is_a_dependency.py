"""What an existence probe answered is an input of the call, yes or no.

Found while stress-testing the decorator: only ``os.path.exists`` and
``isfile`` answering False were recorded. ``Path("cfg.yaml").exists()``
answering False recorded nothing (the ``Path.stat`` wrapper records only a
stat that succeeded), ``os.path.isdir``, ``lexists`` and ``os.access`` were
not watched at all, and a True answer was never recorded, on the assumption
that a read follows -- which is false for a flag file or an output folder.
All of them were served stale after the file appeared or went away.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from cash import Cash


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


PROBES = {
    "Path.exists": lambda p: Path(p).exists(),
    "Path.is_file": lambda p: Path(p).is_file(),
    "os.path.exists": lambda p: os.path.exists(p),
    "os.path.isfile": lambda p: os.path.isfile(p),
    "os.path.lexists": lambda p: os.path.lexists(p),
    "os.access": lambda p: os.access(p, os.F_OK),
}

DIR_PROBES = {
    "Path.is_dir": lambda p: Path(p).is_dir(),
    "os.path.isdir": lambda p: os.path.isdir(p),
    "os.path.exists": lambda p: os.path.exists(p),
}


# Each probe is called through the module at call time, as code does: a
# function object stored in a container keeps the unwrapped original.
def _counted(c, probe):
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def check(p):
        runs.append(1)
        return "custom" if probe(p) else "default"

    return check, runs


@pytest.mark.parametrize("name", sorted(PROBES))
def test_a_file_that_appears_recomputes(c, tmp_path, name):
    check, runs = _counted(c, PROBES[name])
    flag = tmp_path / "flag"
    assert check(str(flag)) == "default"
    assert check(str(flag)) == "default"
    assert len(runs) == 1
    flag.write_text("", encoding="utf-8")
    assert check(str(flag)) == "custom"


@pytest.mark.parametrize("name", sorted(PROBES))
def test_a_file_that_goes_away_recomputes(c, tmp_path, name):
    check, runs = _counted(c, PROBES[name])
    flag = tmp_path / "flag"
    flag.write_text("", encoding="utf-8")
    assert check(str(flag)) == "custom"
    assert check(str(flag)) == "custom"
    assert len(runs) == 1
    flag.unlink()
    assert check(str(flag)) == "default"


@pytest.mark.parametrize("name", sorted(DIR_PROBES))
def test_a_folder_that_appears_and_goes_away_recomputes(c, tmp_path, name):
    check, runs = _counted(c, DIR_PROBES[name])
    out = tmp_path / "out"
    assert check(str(out)) == "default"
    out.mkdir()
    assert check(str(out)) == "custom"
    assert check(str(out)) == "custom"
    assert len(runs) == 2
    shutil.rmtree(out)
    assert check(str(out)) == "default"


def test_a_relative_probe_is_checked_where_the_call_runs(c, tmp_path, monkeypatch):
    check, _ = _counted(c, lambda p: Path(p).exists())
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "cfg.yaml").write_text("", encoding="utf-8")
    monkeypatch.chdir(a)
    assert check("cfg.yaml") == "custom"
    monkeypatch.chdir(b)
    assert check("cfg.yaml") == "default"


def test_a_file_found_but_not_read_is_not_keyed_by_content(c, tmp_path):
    """Being there is what the probe asked; an edit to the flag is not a change."""
    check, runs = _counted(c, lambda p: os.path.exists(p))
    flag = tmp_path / "flag"
    flag.write_text("a", encoding="utf-8")
    assert check(str(flag)) == "custom"
    flag.write_text("something else", encoding="utf-8")
    assert check(str(flag)) == "custom"
    assert len(runs) == 1


def test_writing_into_a_folder_the_call_made_still_hits(c, tmp_path):
    """``os.makedirs(exist_ok=True)`` probes the parents; the call's own
    output there must not make the next call miss."""
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def export(root):
        runs.append(1)
        out = os.path.join(root, "out", "run")
        os.makedirs(out, exist_ok=True)
        return os.path.isdir(out)

    assert export(str(tmp_path)) is True
    assert export(str(tmp_path)) is True
    assert len(runs) == 1


def test_a_probe_answer_reaches_an_enclosing_cached_call(c, tmp_path):
    """Recorded on the inner call, found again from its entry on a hit."""
    flag = tmp_path / "flag"
    check, _ = _counted(c, lambda p: os.path.exists(p))
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def outer(p):
        runs.append(1)
        return check(p) + "!"

    flag.write_text("", encoding="utf-8")
    assert check(str(flag)) == "custom"  # stored first, so outer's inner call hits
    assert outer(str(flag)) == "custom!"
    flag.unlink()
    assert outer(str(flag)) == "default!"
    assert len(runs) == 2
