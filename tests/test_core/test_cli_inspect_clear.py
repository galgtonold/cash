"""Per-function inspection and clearing from the CLI.

A user with a 1.68 GB cache and no disk had two options: keep all of it or
delete all of it. What he wanted was to drop the one function he was finished
with -- and to see which function was responsible before deciding. Neither was
possible: ``cash inspect`` reported a file-extension histogram, and
``cash clear`` took a path or ``--all``.

The grouping needs no new metadata. A decorator cache key is
``{module.qualname}:{state}:{dynamic}:{args}``, so the owning function is
already the first segment of every key on disk.

The awkward part is what a user has to TYPE. A function defined in the script
you ran belongs to module ``__main__``, so its full name is ``__main__.work``
-- which is why an unambiguous trailing segment resolves, and why ambiguity
has to report candidates instead of picking one.
"""
from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

from cash.__main__ import _function_of, _resolve_function, _scan_entries


def _write_entry(cache_dir, stem, key, payload=b"x" * 100):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{stem}.meta").write_bytes(
        pickle.dumps({"key": key, "created_at": 0, "outputs": []}))
    (cache_dir / f"{stem}.data").write_bytes(payload)


def _cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "cash", *args],
                          capture_output=True, text=True, cwd=str(cwd),
                          encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Reading the owner out of a key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "owner"), [
    ("mod.work:aaa:bbb:ccc", "mod.work"),
    ("__main__.work:aaa::ccc", "__main__.work"),
    ("pkg.mod.Klass.method:a:b:c", "pkg.mod.Klass.method"),
    # Notebook statements have no function to name. Reporting them as a
    # function called "stmt" would be worse than grouping them.
    ("stmt:9f8e7d", "(notebook statements)"),
    ("", "(unknown)"),
])
def test_the_owning_function_comes_straight_off_the_key(key, owner):
    assert _function_of(key) == owner


# ---------------------------------------------------------------------------
# Resolving what the user typed
# ---------------------------------------------------------------------------


def _entries(*names):
    return [type("E", (), {"function": n, "size": 1, "mtime": 0.0,
                           "stem": "s", "key": n})() for n in names]


def test_a_trailing_segment_is_enough_when_unambiguous(capsys):
    """Nobody wants to type ``__main__.``."""
    assert _resolve_function(_entries("__main__.work", "lib.other"), "work") \
        == "__main__.work"


def test_an_exact_name_always_wins():
    entries = _entries("work", "pkg.work")
    assert _resolve_function(entries, "work") == "work"


def test_ambiguity_reports_candidates_rather_than_guessing(capsys):
    """The two remedies are not interchangeable, so picking one is not an option."""
    assert _resolve_function(_entries("a.process", "b.process"), "process") is None
    out = capsys.readouterr().out
    assert "ambiguous" in out
    assert "a.process" in out and "b.process" in out


def test_an_unknown_name_lists_what_is_there(capsys):
    assert _resolve_function(_entries("a.process"), "nope") is None
    out = capsys.readouterr().out
    assert "No cached function matches" in out
    assert "a.process" in out, "telling the user they are wrong without saying what is right"


# ---------------------------------------------------------------------------
# Scanning and the commands
# ---------------------------------------------------------------------------


def test_an_entry_is_sized_by_meta_plus_data(tmp_path):
    cache = tmp_path / ".cash"
    _write_entry(cache, "aa", "mod.f:1:2:3", payload=b"y" * 500)
    entry, = _scan_entries(cache)
    assert entry.function == "mod.f"
    assert entry.size > 500, "the payload was not counted"


def test_unreadable_metadata_is_skipped_not_fatal(tmp_path):
    """One corrupt file must not cost you the report for everything else."""
    cache = tmp_path / ".cash"
    _write_entry(cache, "good", "mod.f:1:2:3")
    (cache / "bad.meta").write_bytes(b"not a pickle")
    assert [e.function for e in _scan_entries(cache)] == ["mod.f"]


def test_inspect_groups_by_function(tmp_path):
    cache = tmp_path / ".cash"
    _write_entry(cache, "a1", "mod.big:1:2:3", payload=b"x" * 4000)
    _write_entry(cache, "a2", "mod.big:9:2:3", payload=b"x" * 4000)
    _write_entry(cache, "b1", "mod.small:1:2:3", payload=b"x" * 10)
    out = _cli("inspect", str(cache), cwd=tmp_path).stdout
    assert "mod.big" in out and "mod.small" in out
    # Size-descending: the thing filling the disk is the thing you came to find.
    assert out.index("mod.big") < out.index("mod.small")
    assert "Functions: 2" in out


def test_clear_function_removes_only_that_function(tmp_path):
    cache = tmp_path / ".cash"
    _write_entry(cache, "a1", "mod.doomed:1:2:3")
    _write_entry(cache, "b1", "mod.kept:1:2:3")
    result = _cli("clear", str(cache), "--function", "mod.doomed", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Cleared 1 entry" in result.stdout, "pluralisation"
    remaining = {e.function for e in _scan_entries(cache)}
    assert remaining == {"mod.kept"}
    assert not (cache / "a1.data").exists(), "the payload outlived its metadata"


def test_clear_with_no_arguments_prints_help(tmp_path):
    """It used to name two of the three options and leave you to guess."""
    result = _cli("clear", cwd=tmp_path)
    assert result.returncode == 2
    assert "usage: cash clear" in result.stdout
    assert "--function" in result.stdout


def test_cli_output_is_ascii(tmp_path):
    """A cp1252 console turns an em-dash into a question mark.

    Most of these users are on Windows, where that is the default console
    encoding, so non-ASCII in CLI output is mojibake rather than typography.
    """
    cache = tmp_path / ".cash"
    _write_entry(cache, "a1", "mod.f:1:2:3")
    for argv in (["inspect", str(cache)],
                 ["inspect", str(cache), "--function", "mod.f"],
                 ["clear", str(cache), "--function", "mod.f"]):
        out = _cli(*argv, cwd=tmp_path).stdout
        assert out.isascii(), f"non-ascii in `cash {' '.join(argv)}` output"
