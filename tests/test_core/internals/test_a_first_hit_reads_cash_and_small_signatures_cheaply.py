"""What a process's first cached call reads besides the call's own entry.

The store of analysis answers lives in a folder named for cash's own code,
the size and mtime of each of its ``.py`` files (`scan_store`): a stat of
each, 418 files, was 17 ms of the first call on Windows. Each folder is now
listed once and the stats taken from the listing, which Windows answers
without opening the files; the folder's name is unchanged.

The names a helper's annotations read (`hit_memo._signature_names`) were
looked up in a store whose file, read whole on the first lookup, holds every
answer every earlier process worked out; a ``def`` of a few lines parses in
tens of microseconds, so a small one is parsed instead while that file has
not been read, to the same answer.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import cash
from cash.analysis import scan_store, text_store
from cash.decorator import hit_memo


def _walked_fingerprint() -> str:
    """The fingerprint as ``os.walk`` and a stat of each file gave it."""
    root = Path(cash.__file__).parent
    h = hashlib.blake2b(digest_size=16)
    h.update(f"1\0{getattr(cash, '__version__', '')}\0{sys.version}\0{sys.implementation.cache_tag}".encode())
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                st = os.stat(path)
                entries.append(f"{os.path.relpath(path, root)}\0{st.st_size}\0{st.st_mtime_ns}")
    for entry in sorted(entries):
        h.update(entry.encode())
        h.update(b"\n")
    return h.hexdigest()


def test_cash_is_fingerprinted_from_listings_to_the_same_name(monkeypatch):
    monkeypatch.setattr(scan_store, "_FINGERPRINT", [])
    root = os.path.normcase(os.path.dirname(cash.__file__))
    statted = []
    real = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: statted.append(os.fspath(p)) or real(p, *a, **k))
    found = scan_store._cash_fingerprint()
    monkeypatch.undo()
    mine = [p for p in statted if os.path.normcase(os.path.abspath(p)).startswith(root)]
    assert not mine, mine[:5]
    assert found == _walked_fingerprint()


def test_a_small_signature_is_parsed_without_reading_the_store(monkeypatch):
    monkeypatch.setattr(text_store, "_SLOTS", {})
    monkeypatch.setattr(text_store, "_PENDING", {})
    monkeypatch.setattr(text_store, "_USED", {})
    monkeypatch.setattr(hit_memo, "_PARSED_BUDGET", [10_000])
    looked_up = []
    real = text_store.kept
    monkeypatch.setattr(text_store, "kept", lambda kind, *a: looked_up.append(kind) or real(kind, *a))
    source = "def first_hit_small(a: Alpha, *b: 'Beta') -> Gamma:\n    return a\n"
    assert hit_memo._signature_names(source) == frozenset({"Alpha", "Beta", "Gamma"})
    assert looked_up == []
    assert hit_memo._signature_names(source) == frozenset(hit_memo._parse_signature_names(source))
    large = "def first_hit_large(a: Delta) -> None:\n" + "    x = 1\n" * 200
    assert hit_memo._signature_names(large) == frozenset({"Delta"})
    assert looked_up == ["signature-names"]


def test_small_signatures_go_to_the_store_once_its_file_is_read(monkeypatch):
    monkeypatch.setattr(text_store, "_SLOTS", {"signature-names": {}})
    monkeypatch.setattr(text_store, "_PENDING", {})
    monkeypatch.setattr(text_store, "_USED", {})
    looked_up = []
    real = text_store.kept
    monkeypatch.setattr(text_store, "kept", lambda kind, *a: looked_up.append(kind) or real(kind, *a))
    source = "def first_hit_held(a: Epsilon):\n    return a\n"
    assert hit_memo._signature_names(source) == frozenset({"Epsilon"})
    assert looked_up == ["signature-names"]
