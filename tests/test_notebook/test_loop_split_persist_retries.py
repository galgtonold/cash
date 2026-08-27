"""A loop-split verdict must survive a destination that is briefly locked (#74).

``LoopSplitStore._persist`` wrote a tmp file and called ``os.replace``. On
Windows that call is DENIED, not delayed, while any handle has the
destination open -- and the surrounding ``except OSError`` swallowed it at
debug level. Measured susceptible with a single reader handle held open:

    with a reader handle open: on-disk splits={'hash-a': 4}
       verdict 'hash-b' persisted: False
       in-memory has it anyway   : True
    control (no handle held), 'hash-c' persisted: True

The severity is higher than "a lost optimisation", which is how #74 was
originally filed. The verdict stays in memory but vanishes from disk, so the
NEXT session loads a store without it, declines to split the loop, and keys
the tail differently -- a cache miss and a real recompute of user work. That
is precisely the failure ``LoopSplitStore`` exists to prevent; its own
docstring says a ``k`` that moves between runs changes the tail's key.

Tested by making ``os.replace`` fail a bounded number of times rather than by
holding a real handle on a timer: the retry budget is ~315ms and a
thread-and-sleep test would be a wall-clock threshold, which is this repo's
entire integration-flake class. The real-handle behaviour is pinned once,
Windows-only, at the bottom.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cash.notebook.loop_split import LoopSplitStore
from cash.utils import replace_with_retry


class _DeniesThenSucceeds:
    """``os.replace`` that raises PermissionError *n* times, then works."""

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.calls = 0
        self._real = os.replace

    def __call__(self, src, dst):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise PermissionError(5, "Access is denied")
        return self._real(src, dst)


def test_the_helper_waits_out_a_destination_that_frees_up(tmp_path, monkeypatch):
    target, tmp = tmp_path / "t.json", tmp_path / "t.json.tmp"
    target.write_text("old", encoding="utf-8")
    tmp.write_text("new", encoding="utf-8")

    fake = _DeniesThenSucceeds(failures=2)
    monkeypatch.setattr(os, "replace", fake)
    replace_with_retry(str(tmp), str(target))

    assert target.read_text(encoding="utf-8") == "new"
    assert fake.calls == 3, "expected two denials then a success"


def test_a_persistent_denial_still_raises(tmp_path, monkeypatch):
    """Waiting out contention must not paper over a real permission problem."""
    target, tmp = tmp_path / "t.json", tmp_path / "t.json.tmp"
    target.write_text("old", encoding="utf-8")
    tmp.write_text("new", encoding="utf-8")

    monkeypatch.setattr(os, "replace", _DeniesThenSucceeds(failures=10_000))
    with pytest.raises(PermissionError):
        replace_with_retry(str(tmp), str(target))


def test_a_verdict_survives_a_destination_that_frees_up(tmp_path, monkeypatch):
    """The #74 regression: the verdict must reach disk, not just memory."""
    store = LoopSplitStore(str(tmp_path))
    store.record("hash-a", 4)                     # establishes the file

    monkeypatch.setattr(os, "replace", _DeniesThenSucceeds(failures=2))
    LoopSplitStore(str(tmp_path)).record("hash-b", 7)

    on_disk = json.loads(Path(store._path).read_text(encoding="utf-8"))["splits"]
    assert "hash-b" in on_disk, (
        f"verdict lost on disk despite the destination freeing up: {on_disk}. "
        f"The next session would not split this loop, would key its tail "
        f"differently, and would recompute it."
    )


def test_a_verdict_lost_to_a_permanent_denial_does_not_raise(tmp_path, monkeypatch):
    """Best-effort is still the contract: degrade, never break the cell.

    The store is documented as best-effort throughout -- "the failure mode
    must be 'no optimisation', never 'wrong answer'". Retrying must not turn
    a swallowed persist into an exception in the user's face.
    """
    store = LoopSplitStore(str(tmp_path))
    store.record("hash-a", 4)

    monkeypatch.setattr(os, "replace", _DeniesThenSucceeds(failures=10_000))
    LoopSplitStore(str(tmp_path)).record("hash-b", 7)      # must not raise

    on_disk = json.loads(Path(store._path).read_text(encoding="utf-8"))["splits"]
    assert "hash-b" not in on_disk, "a permanent denial should not have landed"
    assert not list(tmp_path.glob("*.tmp")), "the tmp file was left behind"


@pytest.mark.skipif(os.name != "nt", reason="POSIX never denies a replace")
def test_windows_really_does_deny_a_held_destination(tmp_path):
    """Pins the OS behaviour the retry exists for, so it is evidence, not lore.

    A handle held for the WHOLE call outlasts the retry budget by
    construction, so this asserts the graceful-degradation path rather than
    the recovery one.
    """
    store = LoopSplitStore(str(tmp_path))
    store.record("hash-a", 4)

    with open(store._path, encoding="utf-8"):
        LoopSplitStore(str(tmp_path)).record("hash-b", 7)   # must not raise

    on_disk = json.loads(Path(store._path).read_text(encoding="utf-8"))["splits"]
    assert "hash-a" in on_disk
    assert "hash-b" not in on_disk, (
        "Windows allowed a replace over an open destination -- if this ever "
        "fails, the platform behaviour changed and the retry may be moot"
    )
