"""A sub-unit hit must contribute back into the statement's ambient capture (CAS-243 Task 7).

The call runs *inside* the statement's ambient ``FileAccessTracker`` / output
capture. A genuine miss records the call's effects there for free; a hit
does not, because the callee never runs. ``CallUnit`` closes that gap on a
hit by replaying what the ORIGINAL execution observed:

* ``_replay_deps`` re-declares the entry's recorded file/remote reads onto
  the ambient tracker (``_active_tracker``), mirroring ``core.py``'s
  ``_propagate_file_deps_to_active_tracker`` -- the ``@cash.cache``
  decorator's own defence against exactly this failure mode.
* ``_replay_output`` writes the entry's recorded stdout/stderr onto the
  live stream, reconstructing ``print(a); f(x); print(b)``'s interleaving.
* A call's own cache KEY carries no file content (only source + argument
  lineage), so ``_lookup`` also re-validates ``auto_file_deps`` before
  calling anything a hit -- otherwise a call whose underlying file changed
  would be served forever, and the replay above would just make the
  statement re-declare a staleness nobody underneath it ever notices.

**A subtlety found while writing these tests**: re-validating freshness for
a LOCAL file (``file_dep_is_fresh`` -> ``file_content_hash`` -> ``open()``)
itself performs a real, tracked read through the SAME monkey-patched
``open`` the ambient tracker observes -- so for local paths, the freshness
re-check's own side effect already re-registers the dependency, independent
of ``_replay_deps``. That is harmless (it mirrors the decorator's own
``_auto_file_deps_fresh`` -> ``_propagate_file_deps_to_active_tracker``
ordering, which has the identical property), but it means a LOCAL-only
scenario cannot isolate ``_replay_deps``'s specific contribution. The
REMOTE channel has no such side effect (``remote_dep_is_fresh`` only asks
the store's validator, no local I/O), so that is where this file's
propagation-specific test lives -- and per the task brief, remote deps are
the channel most likely to be skipped by an incomplete implementation.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

import cash
from cash.notebook.call_interception import CallCache, CallSite
from cash.notebook.call_unit import CallUnit
from cash.notebook.file_tracker import FileAccessTracker, _active_tracker
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


# ---------------------------------------------------------------------------
# Channel 1: file deps -- the call's OWN freshness must be re-checked.
# ---------------------------------------------------------------------------

def test_call_hit_recomputes_when_its_own_file_dependency_goes_stale(call_unit_harness, tmp_path):
    """A call's cache KEY never encodes file content, so a stored entry
    whose recorded file changed must not be served forever.

    One-line mutation that breaks this test: in ``CallUnit._lookup``, delete
    the ``if not self._auto_file_deps_fresh(metadata): return False, None,
    0.0, {}`` guard. Applied and observed: the third call returns the STALE
    ``20`` instead of the freshly-computed ``200`` -- verified below, then
    reverted.
    """
    data_path = tmp_path / "data.csv"
    data_path.write_text("10")
    calls: list[int] = []

    def load(k):
        calls.append(k)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return int(data_path.read_text()) * k

    unit = call_unit_harness(lineage={"k": "hash-2"}, user_ns={"k": 2, "load": load})
    site = CallSite(source="load(k)", free_names=frozenset({"load", "k"}), occurrence_index=0)
    wrapped = unit.wrap(load, site)

    # First call: a real miss, inside an ambient tracker (as it would be
    # inside a statement's own execution window).
    with FileAccessTracker() as tracker:
        assert wrapped(2) == 20
    assert calls == [2]

    # Second call, same key -> HIT. The body must not re-run.
    with FileAccessTracker() as tracker:
        assert wrapped(2) == 20
    assert calls == [2], "a fresh hit re-ran the callee"

    # The file changes on disk. The call's key is UNCHANGED (same source,
    # same k), so only its own freshness re-check can catch this.
    data_path.write_text("100")
    with FileAccessTracker() as tracker:
        assert wrapped(2) == 200
    assert calls == [2, 2], (
        "a call whose file dependency went stale was still served the old value"
    )


def test_replay_deps_registers_a_local_path_on_the_ambient_tracker(call_unit_harness):
    """Direct unit test of ``_replay_deps`` for the local-file shape.

    Exercises the method itself (not the full ``_lookup`` round-trip, whose
    freshness re-check has its own tracking side effect for local paths --
    see the module docstring) so the local-path branch is still verified in
    isolation.

    One-line mutation: replace the ``for path, recorded in snap.items():``
    loop body with ``pass``. Applied and observed: ``tracker`` ends up empty
    instead of containing the path -- verified below, then reverted.
    """
    unit = call_unit_harness(lineage={}, user_ns={})
    metadata = {
        "auto_file_deps": {
            "C:/data/input.csv": {"mtime": 1.0, "size": 5, "hash": "deadbeef"},
        },
    }
    with FileAccessTracker() as tracker:
        unit._replay_deps(metadata)
        assert "C:/data/input.csv" in tracker.get_accessed_files()


def test_two_reads_of_the_same_path_in_one_tracker_window_both_stay_correct(call_unit_harness, tmp_path):
    """Regression (coordinator review, round 2): a delta against the AMBIENT
    tracker silently loses a dependency the moment the same path is read
    twice inside one ``FileAccessTracker`` window.

    The original implementation computed ``after - before`` against
    ``_active_tracker.get()`` -- the SHARED tracker for the whole statement
    (or, inside a loop, the whole loop-as-one-unit execution: this is the
    call unit's PRIMARY use case, and the exact shape
    ``call_interception.py``'s own module docstring uses as its running
    example, ``s += compute(x)``). The first call in a window puts the path
    into that shared set; a SECOND call reading the SAME path then computes
    an EMPTY delta against it, even though it genuinely read the file this
    time -- so its own entry stores no ``auto_file_deps`` at all, and
    ``_auto_file_deps_fresh`` (``if not snap: return True``) is vacuously
    fresh forever.

    **Why this is immune to the ``open()``-side-effect masking documented in
    the module docstring above.** That masking only hides a DISABLED REPLAY
    on a call whose ``auto_file_deps`` snapshot was already recorded
    correctly -- the freshness re-check's own ``open()`` call re-registers a
    dependency that already exists in the metadata. Here the bug is
    upstream of that: under the old delta-based code the second call's
    ``auto_file_deps`` is never written in the first place (``_store``'s
    ``if file_deps or remote_deps:`` guard sees an empty set and skips it
    entirely), so ``_auto_file_deps_fresh`` returns ``True`` on the
    ``if not snap`` fast path *before* it ever calls ``file_dep_is_fresh`` /
    ``open()``. No real file read happens during the stale lookup, so there
    is no side effect to mask the bug -- the second call simply serves the
    wrong number.

    One-line mutation that resurrects the bug: in ``wrap``, replace
    ``call_tracker.get_accessed_files()`` / ``get_accessed_remote_urls()``
    (the fresh, per-call tracker's own sets) with a before/after diff against
    ``_active_tracker.get()`` -- i.e. revert to the pre-fix delta. Applied
    and observed: after the file changes, the SECOND call in the shared
    window (``k=20``) still returns the stale ``200`` instead of the
    recomputed ``2000``, while the first call (``k=10``) correctly recomputes
    -- verified below, then reverted.
    """
    data_path = tmp_path / "data.csv"
    data_path.write_text("10")
    calls: list[int] = []

    def expensive(k):
        calls.append(k)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return int(data_path.read_text()) * k

    unit = call_unit_harness(lineage={}, user_ns={"expensive": expensive})
    site = CallSite(
        source="expensive(k)", free_names=frozenset({"expensive", "k"}),
        occurrence_index=0, computed_arg_positions=(0,),
    )
    wrapped = unit.wrap(expensive, site)

    # ONE shared FileAccessTracker window covering BOTH calls -- the shape a
    # loop body actually executes under (each iteration's statement runs
    # inside the SAME tracker instance the loop-as-one-unit fast path
    # entered once for the whole loop), and equally the shape of a single
    # statement that reads a path directly and also calls a cached function
    # that reads it again (``hdr = read(p); total = expensive(k)``).
    with FileAccessTracker():
        assert wrapped(10) == 100   # first read of data.csv this window
        assert wrapped(20) == 200   # second read of the SAME path
    assert calls == [10, 20]

    # The file changes. Re-run both calls, again sharing one window, as a
    # re-executed loop tail would.
    data_path.write_text("100")
    with FileAccessTracker():
        result0 = wrapped(10)
        result1 = wrapped(20)
    assert [result0, result1] == [1000, 2000], (
        "a call whose file dependency changed was served a stale value "
        "because an earlier call in the SAME tracker window had already "
        f"read the same path: got {[result0, result1]}"
    )


# ---------------------------------------------------------------------------
# Channel 2: accessed_remote -- explicitly called out in the brief as the
# channel a first draft skipped entirely. Routed through CallCache.resolve +
# set_sites (not just CallUnit.wrap in isolation), and with no local-file
# side effect to mask the propagation this task adds.
# ---------------------------------------------------------------------------

def test_call_hit_propagates_remote_dependency_through_resolve(tmp_path, monkeypatch):
    """A call that reads a remote object must carry that dependency onto the
    statement's ambient tracker on a HIT, and must recompute when the
    store's own validator reports the object changed.

    One-line mutation that breaks the propagation half: in
    ``CallUnit._replay_deps``, change ``for path, recorded in snap.items():``
    to iterate an empty ``()`` instead. Applied and observed: the second
    call's ambient tracker never sees the URL (assertion on ``tracker2``
    fails) -- verified below, then reverted.
    """
    import cash.remote_source as remote_source

    token = {"value": "etag-v1"}
    monkeypatch.setattr(
        remote_source.RemoteFileDataSource, "state_token",
        lambda self: token["value"],
    )

    url = "s3://bucket/key.csv"
    calls: list[str] = []

    def fetch(u):
        calls.append(u)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        # What a real registered remote-reader handler does: tell the
        # ACTIVE tracker about the read. `_track_path` routes a URL-shaped
        # string to the remote channel on its own (see file_tracker.py).
        tracker = _active_tracker.get()
        if tracker is not None:
            tracker._track_path(u)
        return f"DATA-{token['value']}"

    call_cache = CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))
    site = CallSite(
        source="fetch(u)", free_names=frozenset({"fetch", "u"}), occurrence_index=0,
        computed_arg_positions=(0,),
    )

    call_cache.set_sites([site])
    wrapped1 = call_cache.resolve(fetch)
    with FileAccessTracker() as tracker1:
        assert wrapped1(url) == "DATA-etag-v1"
    assert url in tracker1.get_accessed_remote_urls()
    assert calls == [url]

    # Second call: same site, same url -> the CALL hits (object unchanged).
    call_cache.set_sites([site])
    wrapped2 = call_cache.resolve(fetch)
    with FileAccessTracker() as tracker2:
        assert wrapped2(url) == "DATA-etag-v1"
    assert calls == [url], "a fresh hit re-ran the callee"
    assert url in tracker2.get_accessed_remote_urls(), (
        "a call HIT did not propagate its remote dependency to the ambient tracker"
    )

    # The object changes. The call's key is unchanged (same source, same
    # url) -- only the remote validator re-check can catch this.
    token["value"] = "etag-v2"
    call_cache.set_sites([site])
    wrapped3 = call_cache.resolve(fetch)
    with FileAccessTracker() as tracker3:
        assert wrapped3(url) == "DATA-etag-v2"
    assert calls == [url, url], (
        "a call whose remote dependency changed was still served the old value"
    )


# ---------------------------------------------------------------------------
# Channel 3: stdout / stderr -- replayed onto the live stream on a hit, so
# `print(a); f(x); print(b)` reconstructs its interleaving without the
# statement path needing to know sub-call caching exists.
# ---------------------------------------------------------------------------

def test_call_hit_replays_stdout_reconstructing_interleaving(call_unit_harness, capsys):
    """One-line mutation: in ``wrap``'s hit branch, delete the
    ``self._replay_output(metadata)`` call. Applied and observed: the
    second capture is missing the callee's own line (``"inside=5"``) --
    verified below, then reverted.
    """
    def loud(x):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        print(f"inside={x}")
        return x * 2

    unit = call_unit_harness(lineage={}, user_ns={"loud": loud})
    site = CallSite(
        source="loud(x)", free_names=frozenset({"loud", "x"}), occurrence_index=0,
        computed_arg_positions=(0,),
    )
    wrapped = unit.wrap(loud, site)

    print("before")
    assert wrapped(5) == 10
    print("after")
    out1 = capsys.readouterr().out
    assert out1 == "before\ninside=5\nafter\n"

    # Second call: HIT. `loud` itself does not run, but its recorded print
    # must still appear, interleaved exactly where the call sits.
    print("before2")
    assert wrapped(5) == 10
    print("after2")
    out2 = capsys.readouterr().out
    assert out2 == "before2\ninside=5\nafter2\n", (
        "the callee's stdout was not replayed on a cache hit"
    )


def test_forwarding_tee_records_writelines_not_just_write():
    """Minor (coordinator review): ``__getattr__`` used to delegate
    ``writelines`` straight to the real stream, bypassing ``_chunks`` -- a
    callee using ``sys.stdout.writelines([...])`` produced empty replay text
    on a later hit instead of its actual output.

    One-line mutation: delete the ``writelines`` method (falling back to
    ``__getattr__``'s passthrough). Applied and observed:
    ``tee.getvalue()`` is ``""`` instead of ``"ab"`` -- verified below, then
    reverted.
    """
    import io

    from cash.notebook.call_unit import _ForwardingTee

    real = io.StringIO()
    tee = _ForwardingTee(real)
    tee.writelines(["a", "b"])
    assert tee.getvalue() == "ab"
    assert real.getvalue() == "ab"
