"""After a repair, the check names exactly the variables it recorded again.

``NotebookSimulator.resync_after_replay`` brings the simulation's snapshots
in line with what the repair re-ran or restored, and only those: syncing a
variable the repair did not touch launders a stale value into a match. It
used to find them by comparing every variable the notebook binds, before and
after, on every cell run. A watch on the lineage store (`LineageWatch`) now
notes the few names written; these pin that it names the same ones.
"""

from __future__ import annotations

import copy
import gc
import pickle

from cash.notebook.lineage_store import InputLineages
from cash.notebook.tracking_state import TrackingState


def _state() -> TrackingState:
    state = TrackingState()
    for name, lineage in (("a", "la"), ("b", "lb"), ("c", "lc")):
        state.lineage.record(name, lineage)
        state.executed_input_lineages[name] = {"x": "lx"}
    return state


def _rerecorded(state: TrackingState, watch) -> set[str]:
    return watch.rerecorded(state.variable_lineage, state.executed_input_lineages)


def test_nothing_written_names_nothing():
    state = _state()
    watch = state.lineage.watch()
    assert _rerecorded(state, watch) == set()


def test_a_new_lineage_is_named():
    state = _state()
    watch = state.lineage.watch()
    state.lineage.record("a", "la2")
    assert _rerecorded(state, watch) == {"a"}


def test_the_same_lineage_recorded_with_a_new_input_map_is_named():
    """A re-run that came out the same: its input map is a new one."""
    state = _state()
    watch = state.lineage.watch()
    state.lineage.record("b", "lb")
    state.executed_input_lineages["b"] = {"x": "lx"}
    assert _rerecorded(state, watch) == {"b"}


def test_the_same_lineage_and_the_same_map_is_not_named():
    state = _state()
    watch = state.lineage.watch()
    held = state.executed_input_lineages["c"]
    state.lineage.record("c", "lc")
    state.lineage.reset_to("c", "lc")
    state.executed_input_lineages["c"] = held
    assert _rerecorded(state, watch) == set()


def test_a_lineage_changed_and_changed_back_is_not_named():
    state = _state()
    watch = state.lineage.watch()
    state.lineage.record("a", "other")
    state.lineage.reset_to("a", "la")
    assert _rerecorded(state, watch) == set()


def test_a_new_variable_is_named_and_a_forgotten_one_is_not():
    state = _state()
    watch = state.lineage.watch()
    state.lineage.record("d", "ld")
    state.lineage.discard("a")
    state.executed_input_lineages.pop("a")
    assert _rerecorded(state, watch) == {"d"}


def test_an_input_map_dropped_or_added_is_named():
    state = _state()
    state.lineage.record("e", "le")
    watch = state.lineage.watch()
    state.executed_input_lineages.pop("b")
    state.executed_input_lineages.setdefault("e", {"x": "lx"})
    assert _rerecorded(state, watch) == {"b", "e"}


def test_every_way_of_writing_the_input_maps_is_seen():
    state = _state()
    watch = state.lineage.watch()
    state.executed_input_lineages.update({"a": {}})
    state.executed_input_lineages |= {"b": {}}
    del state.executed_input_lineages["c"]
    assert _rerecorded(state, watch) == {"a", "b", "c"}

    watch = state.lineage.watch()
    state.executed_input_lineages.clear()
    assert _rerecorded(state, watch) == {"a", "b"}


def test_a_forgotten_session_names_nothing_it_no_longer_holds():
    state = _state()
    watch = state.lineage.watch()
    state.reset_session_state()
    assert isinstance(state.executed_input_lineages, InputLineages)
    assert _rerecorded(state, watch) == set()
    state.lineage.record("a", "la")
    assert _rerecorded(state, watch) == {"a"}


def test_two_watches_each_see_their_own_stretch():
    """A check inside another (a repair re-running a cell that is checked):
    the outer watch still sees what the inner one saw."""
    state = _state()
    outer = state.lineage.watch()
    state.lineage.record("a", "la2")
    inner = state.lineage.watch()
    state.lineage.record("b", "lb2")
    assert _rerecorded(state, inner) == {"b"}
    assert _rerecorded(state, outer) == {"a", "b"}


def test_a_dropped_watch_stops_collecting():
    state = _state()
    watch = state.lineage.watch()
    del watch
    gc.collect()
    assert len(state.lineage._watches) == 0
    state.lineage.record("a", "la2")
    assert len(state.lineage._watches) == 0


def test_a_copy_of_the_input_maps_is_a_plain_dict():
    state = _state()
    for copied in (copy.copy(state.executed_input_lineages), pickle.loads(pickle.dumps(state.executed_input_lineages))):
        assert type(copied) is dict
        assert copied == state.executed_input_lineages


def test_a_state_built_with_input_maps_is_watched():
    state = TrackingState(executed_input_lineages={"a": {"x": "lx"}})
    state.lineage.record("a", "la")
    watch = state.lineage.watch()
    state.executed_input_lineages["a"] = {"x": "lx"}
    assert _rerecorded(state, watch) == {"a"}
