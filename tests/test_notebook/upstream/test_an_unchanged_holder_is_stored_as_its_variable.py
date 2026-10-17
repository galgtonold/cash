"""An entry keeps an object a variable holds, which the statement left as it
was, as that variable: a hit binds the live object, never a copy.

``gs = {'est': full, ...}`` while a closure (``keep``) holds ``full`` too:
no restore can rebind the closure, so the statement used to run every time
(E6's ``grid_search.fit`` holding the pipelines built earlier). The
statement leaves ``full`` as it was, so the entry stores it as a reference
to the variable (`cash.notebook.anchors`), checked by its content digest.
A restore binds the live ``full`` when that digest still matches, and
misses otherwise: identity as in plain Python, or a run.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.anchors import AnchorRef, resolve_anchors, with_anchor_refs
from cash.notebook.cache_status import CacheStatus
from cash.value_hash import budgeted_hash, compute_hash
from tests._cell_driver import run_cash_cell

FIT = "gs = {'est': full, 'w': (time.sleep(0.2), 2)[1]}"
CELLS = [
    "import time",
    "full = {'pre': [1, 2]}",
    "keep = (lambda held: lambda: held)(full)",
    FIT,
]
NAMES = ["time", "full", "keep", "gs", "box"]


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what the disk tier kept."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _status(magics, code: str) -> str | None:
    statements = magics.cash_status("dict")["last_cell"]["statements"]
    return next((m.get("status") for m in statements if m.get("code") == code), None)


def _run(magics, cells: list[str], notebook: list[str] | None = None) -> str | None:
    """Run *cells* of *notebook* (by default, *cells* are the notebook); the
    status of the last one's statement."""
    for cell in cells:
        run_cash_cell(magics, cell, cells=notebook or cells)
    return _status(magics, cells[-1])


def _same_as_plain(ns) -> None:
    assert ns["gs"] == {"est": {"pre": [1, 2]}, "w": 2}
    assert ns["gs"]["est"] is ns["full"], "the live object, not a copy"
    assert ns["keep"]() is ns["gs"]["est"]
    ns["full"]["pre"].append(3)
    assert ns["gs"]["est"]["pre"] == [1, 2, 3], "a change through the variable shows in the output"


def test_a_hit_in_the_same_session_binds_the_same_object(cash_magics, clean_backend):
    assert _run(cash_magics, CELLS) == CacheStatus.COMPUTED

    assert _run(cash_magics, [FIT], CELLS) == CacheStatus.RESTORED
    _same_as_plain(cash_magics.shell.user_ns)


def test_after_a_restart_the_rebuilt_variable_is_bound(cash_magics, clean_backend):
    _run(cash_magics, CELLS)
    _restart(cash_magics)

    assert _run(cash_magics, CELLS) == CacheStatus.RESTORED
    _same_as_plain(cash_magics.shell.user_ns)


def test_after_a_restart_a_variable_built_otherwise_is_a_miss(cash_magics, clean_backend):
    """The statement reads ``box``, which holds ``full``'s dict; ``full`` is
    the anchor. After the restart ``full`` is bound again, to an equal dict,
    after ``box`` was built: the entry's key and the anchor's digest match,
    but ``box`` holds the other dict, which a run gives the outputs. The
    statement runs."""
    fit = "gs = {'est': box['full'], 'w': (time.sleep(0.2), 2)[1]}"
    cells = [*CELLS[:3], "box = {'full': full}", fit]
    assert _run(cash_magics, cells) == CacheStatus.COMPUTED
    _restart(cash_magics)

    rebuilt = [*cells[:-1], "full = {'pre': [1] + [2]}", fit]
    assert _run(cash_magics, rebuilt) != CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert ns["gs"]["est"] is ns["box"]["full"]
    assert ns["gs"]["est"] is not ns["full"]


def test_a_variable_no_longer_holding_the_anchor_is_a_miss(cash_magics, clean_backend):
    """``full`` bound to an equal new dict: its digest matches, but ``box``
    holds the old one, which is the one the statement reads. Binding the
    new ``full`` would serve a different object than a run gives."""
    fit = "gs = {'est': box['full'], 'w': (time.sleep(0.2), 2)[1]}"
    cells = [*CELLS[:3], "box = {'full': full}", fit]
    _run(cash_magics, cells)

    rebound = ["full = {'pre': [1] + [2]}", fit]
    assert _run(cash_magics, rebound, [*cells[:-1], *rebound]) != CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert ns["gs"]["est"] is ns["box"]["full"]
    assert ns["gs"]["est"] is not ns["full"]


def test_a_change_to_the_shared_object_after_storing_runs_again(cash_magics, clean_backend):
    """``full`` changed in place after the entry was stored: the statement
    reads it, so its key moves on, and it runs, as plain Python does."""
    _run(cash_magics, CELLS)

    changed = ["full['pre'].append(3)", FIT]
    assert _run(cash_magics, changed, [*CELLS[:-1], *changed]) == CacheStatus.COMPUTED
    ns = cash_magics.shell.user_ns
    assert ns["gs"]["est"] is ns["full"] is ns["keep"]()
    assert ns["gs"]["est"] == {"pre": [1, 2, 3]}


def test_sharing_inside_the_anchor_is_part_of_its_digest(cash_magics, clean_backend):
    """As for any value (K10): two keys holding one list are not the same
    content as two equal lists. ``full`` rebuilt with two lists after a
    restart keys the statement apart, and its digest is not the anchor's
    either (`test_a_digest_tells_shared_parts_apart`): a miss."""
    shared = [CELLS[0], "part = [1]\nfull = {'a': part, 'b': part}", CELLS[2], FIT]
    _run(cash_magics, shared)
    _restart(cash_magics)

    apart = [CELLS[0], "part = [1]\nfull = {'a': part, 'b': [1]}", CELLS[2], FIT]
    assert _run(cash_magics, apart) != CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert ns["gs"]["est"] is ns["full"]
    assert ns["gs"]["est"]["a"] is not ns["gs"]["est"]["b"]


class TestResolve:
    """`resolve_anchors` on its own."""

    def test_binds_the_live_object_and_keeps_what_the_parts_share(self):
        full = {"pre": [1, 2]}
        value = {"est": full, "again": full, "own": [full]}
        stored = with_anchor_refs(value, {"full": full})
        assert stored["est"] == AnchorRef("full")

        live = {"pre": [1, 2]}
        out = resolve_anchors(stored, {"full": budgeted_hash(full, compute_hash)}, {"full": live})
        assert out["est"] is out["again"] is out["own"][0] is live

    def test_a_changed_variable_raises(self):
        full = {"pre": [1, 2]}
        stored = with_anchor_refs({"est": full}, {"full": full})
        with pytest.raises(KeyError, match="changed"):
            resolve_anchors(stored, {"full": budgeted_hash(full, compute_hash)}, {"full": {"pre": [9]}})

    def test_an_unbound_variable_raises(self):
        full = {"pre": [1, 2]}
        stored = with_anchor_refs({"est": full}, {"full": full})
        with pytest.raises(KeyError, match="not bound"):
            resolve_anchors(stored, {"full": budgeted_hash(full, compute_hash)}, {})

    def test_a_holder_bound_now_must_hold_the_anchor(self):
        full = {"pre": [1, 2]}
        stored = with_anchor_refs({"est": full}, {"full": full})
        digests = {"full": budgeted_hash(full, compute_hash)}
        holds = {"box": ["full"]}
        assert resolve_anchors(stored, digests, {"full": full, "box": {"f": full}}, holds=holds)["est"] is full
        assert resolve_anchors(stored, digests, {"full": full}, holds=holds)["est"] is full, (
            "not bound: nothing to share"
        )
        with pytest.raises(KeyError, match="no longer holds"):
            resolve_anchors(stored, digests, {"full": full, "box": {"f": {"pre": [1, 2]}}}, holds=holds)

    def test_a_digest_tells_shared_parts_apart(self):
        """K10: a mutable part shared inside the anchor is part of its content."""
        part = [1]
        full = {"a": part, "b": part}
        stored = with_anchor_refs({"est": full}, {"full": full})
        digests = {"full": budgeted_hash(full, compute_hash)}
        with pytest.raises(KeyError, match="changed"):
            resolve_anchors(stored, digests, {"full": {"a": [1], "b": [1]}})
        rebuilt = [1]
        assert resolve_anchors(stored, digests, {"full": {"a": rebuilt, "b": rebuilt}})["est"]["a"] is rebuilt
