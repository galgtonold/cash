"""A read outside every cached call is credited to the user code on the stack
(`note_untracked_read`), by the file's resolved path and its stat when read.

Resolving the path (``realpath``, ~60 us on Windows) is put off until
something asks what the code read (`credited_reads`); the stat is still taken
at the read. What must hold: the same files and stats are credited as when
each path was resolved at its read -- for relative, absolute and linked
spellings; for user code above a library; for none of cash's own reads --
and the latest read of a file wins, whichever way it was spelled and
whether it was made inside a cached call or outside.
"""

from __future__ import annotations

import os

import pytest

import cash  # noqa: F401  (its open handler, watching outside cached calls)
import cash.tracking.read_credit as read_credit
from cash._paths import normalize_path
from cash.tracking.file_tracker import FileAccessTracker
from cash.tracking.read_classification import regular_file_stat

_CASH_DIR = os.path.dirname(os.path.abspath(read_credit.__file__))


def _function(source: str, filename: str, name: str, namespace: dict):
    # Code objects alike but for their file compare equal, and reads are
    # kept by code: each test's own, by a constant of its own.
    source = source.replace("):\n", "):\n    %r\n" % filename, 1)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def _resolved(path) -> str:
    return normalize_path(os.path.realpath(os.path.abspath(path)))


def _rewrite(path, text: str) -> None:
    """Change the file, as a second later: a stat taken outside a cached call
    is reused for a second (`note_untracked_read`)."""
    read_credit._untracked_stats.clear()
    before = os.stat(path)
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))


@pytest.fixture
def user_reader(tmp_path):
    """User code that opens each path it is given, and what it read."""
    ns: dict = {}
    reader = _function(
        "def reader(paths):\n    for p in paths:\n        with open(p, encoding='utf-8') as fh:\n            fh.read()\n",
        str(tmp_path / "user_reader.py"),
        "reader",
        ns,
    )
    return reader


@pytest.fixture
def spelled(tmp_path, monkeypatch):
    """One file read three ways where links can be made: relative, absolute, linked."""
    (tmp_path / "data").mkdir()
    target = tmp_path / "data" / "table.txt"
    target.write_text("first", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("other", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    spellings = [os.path.join("data", "table.txt"), str(target)]
    try:
        os.symlink(tmp_path / "data", tmp_path / "link", target_is_directory=True)
        spellings.append(str(tmp_path / "link" / "table.txt"))
    except (OSError, NotImplementedError):
        pass  # Windows without the privilege
    return target, other, spellings


def test_user_code_is_credited_each_file_by_its_resolved_path_and_stat(user_reader, spelled):
    target, other, spellings = spelled
    user_reader([*spellings, str(other)])
    credited = read_credit.credited_reads(user_reader.__code__)
    assert credited == {
        _resolved(target): regular_file_stat(str(target)),
        _resolved(other): regular_file_stat(str(other)),
    }


def test_the_user_code_above_a_library_is_credited_and_the_library_is_not(tmp_path, spelled):
    target, _, _ = spelled
    ns: dict = {}
    ns["load"] = _function(
        "def load(p):\n    with open(p, encoding='utf-8') as fh:\n        return fh.read()\n", "<library>", "load", ns
    )
    user = _function("def user(p):\n    return load(p)\n", str(tmp_path / "user_above.py"), "user", ns)
    user(str(target))
    assert read_credit.credited_reads(user.__code__) == {_resolved(target): regular_file_stat(str(target))}
    assert read_credit.credited_reads(ns["load"].__code__) == {}


def test_a_read_cash_makes_below_user_code_is_nobody_s(tmp_path, spelled):
    target, _, _ = spelled
    ns: dict = {}
    ns["own"] = _function(
        "def own(p):\n    with open(p, encoding='utf-8') as fh:\n        return fh.read()\n",
        os.path.join(_CASH_DIR, "reader_in_this_test.py"),
        "own",
        ns,
    )
    user = _function("def user(p):\n    return own(p)\n", str(tmp_path / "user_over_cash.py"), "user", ns)
    user(str(target))
    assert read_credit.credited_reads(user.__code__) == {}


def test_the_latest_read_wins_whatever_its_spelling(user_reader, spelled):
    target, _, spellings = spelled
    for i, spelling in enumerate([*spellings, spellings[0]]):
        _rewrite(target, "version %d" % i)
        user_reader([spelling])
    assert read_credit.credited_reads(user_reader.__code__) == {_resolved(target): regular_file_stat(str(target))}


@pytest.mark.parametrize("last", ["inside", "outside"])
def test_the_latest_read_wins_inside_a_cached_call_or_outside(user_reader, spelled, last):
    target, _, spellings = spelled

    def inside():
        with FileAccessTracker():
            user_reader([str(target)])

    first, then = (
        (lambda: user_reader([spellings[0]]), inside)
        if last == "inside"
        else (inside, lambda: user_reader([spellings[0]]))
    )
    _rewrite(target, "old")
    first()
    _rewrite(target, "new text")
    then()
    assert read_credit.credited_reads(user_reader.__code__) == {_resolved(target): regular_file_stat(str(target))}


def test_a_link_retargeted_before_the_lookup_credits_the_new_target_with_the_old_stat(user_reader, tmp_path):
    """Pinned: the path is resolved at the lookup, so a link retargeted
    between the read and the lookup names the file it points to then. Before
    reads were resolved late, the file it pointed to at the read was named
    -- as it still is when the path was resolved before (`_untracked_realpaths`
    keeps the first answer). Either way the stat is the one the read took:
    it is not the new target's, so the entry using it is refused, not served
    (`FileDeps.credit_remembered_reads`)."""
    for name, text in (("a", "first target"), ("b", "the other target, longer")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "table.txt").write_text(text, encoding="utf-8")
    link = tmp_path / "current"
    try:
        os.symlink(tmp_path / "a", link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    read_stat = regular_file_stat(str(tmp_path / "a" / "table.txt"))
    user_reader([str(link / "table.txt")])
    os.remove(link) if os.name != "nt" else os.rmdir(link)
    os.symlink(tmp_path / "b", link, target_is_directory=True)
    credited = read_credit.credited_reads(user_reader.__code__)
    assert credited == {_resolved(tmp_path / "b" / "table.txt"): read_stat}
    assert read_stat != regular_file_stat(str(tmp_path / "b" / "table.txt"))
