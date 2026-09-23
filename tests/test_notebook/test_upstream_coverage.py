"""Comprehensive unit tests for upstream.py to improve coverage from 62% to 70%+.

Tests cover:
- MismatchClassifier.classify on a lineage mismatch
- lineage_formula.input_lineage (as the simulator calls it)
- the input lineages a simulated statement is keyed on
- lineage_formula.module_source_component (the simulator's module component)
- _resolve_fallback_cache_idx / _reset_advanced_lineages
- _handle_downstream_advancement_fallback
- _stat_file_deps
- restore_statement's file and lineage checks
- reset_caches
- set_tracking_state
"""

import os
import types
from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest

from cash.notebook._protocols import TrackingState
from cash.notebook.upstream import UpstreamChecker
from cash.notebook.upstream._types import CellCheck, SimulationResult, TraceEntry
from cash.notebook.upstream.virtual_lineage import VirtualLineage, lineage_conflict
from cash.tracking.file_dep_snapshot import snapshot_file_deps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checker(**kwargs):
    """Create an UpstreamChecker with a mock shell."""
    shell = MagicMock()
    shell.user_ns = kwargs.pop("user_ns", {})
    cash_instance = kwargs.pop("cash_instance", None)
    tracking_state = kwargs.pop("tracking_state", None)
    compute_hash_fn = kwargs.pop("compute_hash_fn", None)
    checker = UpstreamChecker(
        shell,
        cash_instance=cash_instance,
        compute_hash_fn=compute_hash_fn,
        tracking_state=tracking_state,
    )
    return checker


# ===========================================================================
# MismatchClassifier.classify: a lineage mismatch
# ===========================================================================


class TestLineageMismatch:
    """A required input whose live lineage differs from the simulated one."""

    def _classify(self, checker, actual="actual", virtual="virtual", trace=None):
        checker.tracking_state.lineage.record("x", actual)
        sim = SimulationResult(virtual_lineage={"x": virtual}, trace=trace or [])
        check = CellCheck(current_cell_idx=1, notebook_cells=["x = 1", "y = x"], required_inputs={"x"})
        return checker.simulator.classifier.classify(sim, check).broken_vars

    def test_mismatch_adds_to_broken_vars(self):
        """When lineage doesn't match, var should be added to broken_vars."""
        checker = _make_checker(user_ns={"x": 1})
        assert "x" in self._classify(checker, actual="aaa", virtual="bbb")

    def test_matching_lineage_not_broken(self):
        """When actual matches virtual, var should not be broken."""
        checker = _make_checker(user_ns={"x": 1})
        assert "x" not in self._classify(checker, actual="same", virtual="same")

    def test_debug_mode_no_crash(self):
        """Debug mode should produce debug output without crashing."""
        checker = _make_checker(user_ns={"x": 1})
        assert "x" in self._classify(checker, actual="expected123", virtual="actual456")

    def test_with_simulation_trace(self):
        """The trace's last producer of the variable is looked up."""
        checker = _make_checker(user_ns={"x": 1})
        trace = [TraceEntry("x = 1", {"x"}, set(), {}, {}, False)]
        assert "x" in self._classify(checker, trace=trace)


# ===========================================================================
# lineage_formula.input_lineage, as the simulator calls it
# ===========================================================================


def _resolve(checker, name, virtual):
    from cash.notebook.lineage_formula import input_lineage

    return input_lineage(
        name,
        checker.shell.user_ns,
        (virtual, checker.variable_lineage),
        compute_hash=checker.compute_hash_fn,
        function_tracker=None,
        code=None,
    )


class TestResolveInputLineage:
    """Test the priority-based input lineage resolution."""

    def test_virtual_lineage_first(self):
        """Virtual lineage should be checked first."""
        checker = _make_checker()
        checker.tracking_state.lineage.record("x", "runtime_hash")
        result = _resolve(checker, "x", {"x": "virtual_hash"})
        assert result == "virtual_hash"

    def test_falls_back_to_variable_lineage(self):
        """Falls back to variable_lineage when not in virtual."""
        checker = _make_checker()
        checker.tracking_state.lineage.record("x", "runtime_hash")
        result = _resolve(checker, "x", {})
        assert result == "runtime_hash"

    def test_falls_back_to_user_ns_hash(self):
        """Falls back to hashing from user_ns when no lineage exists."""
        checker = _make_checker(user_ns={"x": 42})
        result = _resolve(checker, "x", {})
        assert result is not None
        assert len(result) == 64

    def test_custom_compute_hash_fn(self):
        """Should use compute_hash_fn when available."""
        checker = _make_checker(
            user_ns={"x": 42},
            compute_hash_fn=lambda v: "custom_hash_result",
        )
        result = _resolve(checker, "x", {})
        assert result == "custom_hash_result"

    def test_returns_none_for_missing_variable(self):
        """Returns None when variable is not in any source."""
        checker = _make_checker()
        result = _resolve(checker, "x", {})
        assert result is None

    def test_returns_none_for_none_value(self):
        """Returns None when user_ns has None for the variable."""
        checker = _make_checker(user_ns={"x": None})
        result = _resolve(checker, "x", {})
        assert result is None

    def test_a_module_with_no_lineage_contributes_nothing(self):
        """Hashing a module would bake a memory address into the lineage."""
        checker = _make_checker(user_ns={"x": os})
        assert _resolve(checker, "x", {}) is None
        checker.tracking_state.lineage.record("x", "module_lineage")
        assert _resolve(checker, "x", {}) == "module_lineage"


# ===========================================================================
# The input lineages a simulated statement is keyed on
# ===========================================================================


class TestSimulatedInputLineages:
    """What a simulated statement's output lineage is built from."""

    @staticmethod
    def _lineage_of_x(checker, code, virtual=None):
        return checker.simulator.simulate_cell(code, virtual or {}).virtual_lineage["x"]

    def test_every_input_counts(self):
        checker = _make_checker()
        checker.tracking_state.lineage.record("a", "hash_a")
        checker.tracking_state.lineage.record("b", "hash_b")
        before = self._lineage_of_x(checker, "x = a + b")
        checker.tracking_state.lineage.record("b", "hash_b2")
        assert self._lineage_of_x(checker, "x = a + b") != before

    def test_skips_get_ipython(self):
        """``get_ipython`` contributes nothing, whatever it is bound to."""
        checker = _make_checker(user_ns={"get_ipython": object()})
        checker.tracking_state.lineage.record("a", "hash_a")
        before = self._lineage_of_x(checker, "x = a if get_ipython else a")
        checker.shell.user_ns["get_ipython"] = object()
        assert self._lineage_of_x(checker, "x = a if get_ipython else a") == before

    def test_virtual_lineage_priority(self):
        """The simulation's own lineage wins over the recorded one."""
        checker = _make_checker()
        via_virtual = self._lineage_of_x(checker, "x = a", {"a": "virtual_hash"})
        checker.tracking_state.lineage.record("a", "runtime_hash")
        assert self._lineage_of_x(checker, "x = a", {"a": "virtual_hash"}) == via_virtual
        assert self._lineage_of_x(checker, "x = a") != via_virtual

    def test_empty_inputs(self):
        """A statement reading nothing does not depend on what is recorded."""
        checker = _make_checker()
        before = self._lineage_of_x(checker, "x = 1")
        checker.tracking_state.lineage.record("a", "hash_a")
        assert self._lineage_of_x(checker, "x = 1") == before


# ===========================================================================
# _stat_file_deps
# ===========================================================================


class TestStatFileDeps:
    """Test static file dependency stat helper."""

    def test_existing_files(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        result = VirtualLineage._stat_file_deps({str(f): 0.0})
        assert str(f) in result
        assert result[str(f)] == pytest.approx(os.path.getmtime(str(f)), abs=0.1)

    def test_missing_files_excluded(self, tmp_path):
        missing = str(tmp_path / "nonexistent.csv")
        result = VirtualLineage._stat_file_deps({missing: 0.0})
        assert missing not in result

    def test_empty_input(self):
        result = VirtualLineage._stat_file_deps({})
        assert result == {}

    def test_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("data1")
        f2.write_text("data2")
        result = VirtualLineage._stat_file_deps({str(f1): 0.0, str(f2): 0.0})
        assert len(result) == 2


# ===========================================================================
# restore_statement: when a cache entry may be restored
# ===========================================================================


def _restoring_checker(metadata, variables=None):
    backend = MagicMock()
    backend.get.return_value = ({"execution_time": 1.0, **metadata}, {"variables": variables or {"x": 1}})
    return _make_checker(cash_instance=MagicMock(backend=backend))


def _restore(checker, expected_lineages=None):
    return checker.simulator.restore_statement("x = f()", {"x"}, {"f"}, {}, expected_lineages=expected_lineages)


class TestRestoreChecksFileDeps:
    """An entry whose files changed is not restored."""

    def test_all_fresh(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("content")
        checker = _restoring_checker({"file_dependencies": snapshot_file_deps({str(f)})})
        assert _restore(checker) == {"x"}

    def test_stale_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("content")
        snapshot = snapshot_file_deps({str(f)})
        f.write_text("changed content")
        checker = _restoring_checker({"file_dependencies": snapshot})
        assert _restore(checker) == set()

    def test_missing_file(self, tmp_path):
        checker = _restoring_checker({"file_dependencies": {str(tmp_path / "gone.csv"): {"mtime": 1.0}}})
        assert _restore(checker) == set()

    def test_empty_deps(self):
        checker = _restoring_checker({"file_dependencies": {}})
        assert _restore(checker) == {"x"}


class TestLineageConflict:
    """An entry built for other lineages than the simulation expects is not restored."""

    def test_consistent_lineage(self):
        checker = _restoring_checker({"output_lineages": {"x": "hash_x"}})
        assert _restore(checker, {"x": "hash_x"}) == {"x"}

    def test_inconsistent_lineage(self):
        checker = _restoring_checker({"output_lineages": {"x": "cached_hash"}})
        assert _restore(checker, {"x": "expected_hash"}) == set()

    def test_skipped_with_file_deps(self, tmp_path):
        """With file deps the lineages are not compared: the files decide."""
        f = tmp_path / "data.csv"
        f.write_text("content")
        metadata = {"output_lineages": {"x": "cached_hash"}, "file_dependencies": snapshot_file_deps({str(f)})}
        assert lineage_conflict(metadata, metadata["file_dependencies"], {"x": "expected_hash"}) is None
        assert _restore(_restoring_checker(metadata), {"x": "expected_hash"}) == {"x"}

    def test_no_expected_lineages(self):
        checker = _restoring_checker({"output_lineages": {"x": "hash"}})
        assert _restore(checker, None) == {"x"}

    def test_no_output_lineages_in_metadata(self):
        assert lineage_conflict({}, {}, {"x": "hash"}) is None
        assert _restore(_restoring_checker({}), {"x": "hash"}) == {"x"}


class TestBackendLess:
    """Without a cache backend nothing is probed and nothing restored."""

    def test_no_cash_instance(self):
        checker = _make_checker()
        assert _restore(checker) == set()
        assert "x" in checker.simulator.simulate_cell("x = 1").virtual_lineage

    def test_simulation_probes_metadata_only(self):
        backend = MagicMock()
        backend.get_metadata.return_value = None
        checker = _make_checker(cash_instance=MagicMock(backend=backend))
        checker.simulator.simulate_cell("x = 1")
        assert any(str(c.args[0]).startswith("stmt:") for c in backend.get_metadata.call_args_list)


# ===========================================================================
# reset_caches
# ===========================================================================


class TestResetCaches:
    """Test cache clearing."""

    def test_clears_all_caches(self):
        checker = _make_checker()
        checker.simulator.cache.entries.append(MagicMock())
        checker.simulator.cache.cell_hashes[0] = "hash"
        checker.reset_caches()
        assert len(checker.simulator.cache.entries) == 0
        assert len(checker.simulator.cache.cell_hashes) == 0


# ===========================================================================
# set_tracking_state
# ===========================================================================


class TestSetTrackingState:
    """Test tracking state wiring."""

    def test_wires_all_dicts(self):
        state = TrackingState()
        state.lineage.record("x", "hash")
        state.executed_cell_codes["x"] = "x = 1"
        checker = _make_checker(tracking_state=state)
        assert checker.variable_lineage is state.variable_lineage
        assert checker.executed_cell_codes is state.executed_cell_codes
        assert checker.variable_lineage["x"] == "hash"

    def test_new_state_replaces_old(self):
        state1 = TrackingState()
        state1.lineage.record("x", "old")
        state2 = TrackingState()
        state2.lineage.record("y", "new")
        checker = _make_checker(tracking_state=state1)
        checker.set_tracking_state(state2)
        assert "x" not in checker.variable_lineage
        assert checker.variable_lineage["y"] == "new"


# ===========================================================================
# _resolve_fallback_cache_idx
# ===========================================================================


class TestResolveFallbackCacheIdx:
    """Test downstream advancement fallback index resolution."""

    def test_returns_none_with_no_cache(self):
        checker = _make_checker()
        checker.last_cell_index = None
        result = checker._resolve_fallback_cache_idx(None)
        assert result is None

    def test_returns_none_for_cell_at_index_zero(self):
        """Cell at index 0 has no prior cell to fall back to."""
        checker = _make_checker()
        checker.last_cell_index = None
        checker.simulator.cache.last_index_by_cell_id["cell_0"] = 0
        result = checker._resolve_fallback_cache_idx("cell_0")
        assert result is None

    def test_returns_previous_cache_index(self):
        checker = _make_checker()
        checker.last_cell_index = None
        checker.simulator.cache.last_index_by_cell_id["cell_2"] = 2
        # Need at least 2 simulation cache entries
        checker.simulator.cache.entries = [MagicMock(), MagicMock(), MagicMock()]
        result = checker._resolve_fallback_cache_idx("cell_2")
        assert result == 1  # Previous index

    def test_uses_last_cell_index_without_cell_id(self):
        checker = _make_checker()
        checker.last_cell_index = 3
        checker.simulator.cache.entries = [MagicMock()] * 4
        result = checker._resolve_fallback_cache_idx(None)
        assert result == 2

    def test_cache_too_small(self):
        """If simulation cache is smaller than target index, return None."""
        checker = _make_checker()
        checker.last_cell_index = None
        checker.simulator.cache.last_index_by_cell_id["cell_5"] = 5
        checker.simulator.cache.entries = [MagicMock()]  # Only 1 entry
        result = checker._resolve_fallback_cache_idx("cell_5")
        assert result is None


# ===========================================================================
# _reset_advanced_lineages
# ===========================================================================


class TestResetAdvancedLineages:
    """Test resetting lineages that are 'ahead' of cached virtual lineage."""

    def test_resets_mismatched_lineage(self):
        checker = _make_checker()
        checker.tracking_state.lineage.record("x", "runtime_ahead")
        cached_virtual = {"x": "virtual_correct"}
        checker._reset_advanced_lineages({"x"}, cached_virtual, 0)
        assert checker.variable_lineage["x"] == "virtual_correct"

    def test_skips_matching_lineage(self):
        checker = _make_checker()
        checker.tracking_state.lineage.record("x", "same_hash")
        cached_virtual = {"x": "same_hash"}
        checker._reset_advanced_lineages({"x"}, cached_virtual, 0)
        assert checker.variable_lineage["x"] == "same_hash"

    def test_skips_variables_not_in_virtual(self):
        checker = _make_checker()
        checker.tracking_state.lineage.record("x", "runtime_hash")
        checker._reset_advanced_lineages({"x"}, {}, 0)
        assert checker.variable_lineage["x"] == "runtime_hash"

    def test_skips_variables_not_in_runtime(self):
        checker = _make_checker()
        cached_virtual = {"x": "virtual_hash"}
        checker._reset_advanced_lineages({"x"}, cached_virtual, 0)
        # x was not in variable_lineage, should still not be there
        assert "x" not in checker.variable_lineage


# ===========================================================================
# _handle_downstream_advancement_fallback
# ===========================================================================


class TestHandleDownstreamAdvancementFallback:
    """Test downstream advancement fallback logic."""

    def test_no_op_without_simulation_cache(self):
        checker = _make_checker()
        # Should not raise
        checker._handle_downstream_advancement_fallback(cell_id=None, required_inputs={"x"}, current_cell_outputs={"x"})

    def test_no_op_without_overlap(self):
        checker = _make_checker()
        checker.simulator.cache.entries = [MagicMock()]
        checker._handle_downstream_advancement_fallback(cell_id=None, required_inputs={"a"}, current_cell_outputs={"b"})

    def test_no_op_with_empty_inputs(self):
        checker = _make_checker()
        checker.simulator.cache.entries = [MagicMock()]
        checker._handle_downstream_advancement_fallback(cell_id=None, required_inputs=set(), current_cell_outputs={"x"})


# ===========================================================================
# Module source component (shared with the runtime via lineage_formula)
# ===========================================================================


class TestModuleSourceComponent:
    """The simulator's module component is the runtime's: one function."""

    @staticmethod
    def _component(tracker, value, name, code="x = 1"):
        from cash.notebook.lineage_formula import module_source_component

        return module_source_component(tracker, value, name, code)

    def test_no_function_tracker(self):
        assert self._component(None, os, "os") == ""

    def test_non_module_output(self):
        assert self._component(MagicMock(), 42, "x") == ""

    def test_module_not_tracked(self):
        """Module in user_ns but not in function_tracker.tracked_modules."""
        mod = types.ModuleType("fake_mod")
        mod.__file__ = "/nonexistent/fake_mod.py"
        tracker = MagicMock()
        tracker.tracked_modules = set()
        assert self._component(tracker, mod, "fake_mod") == ""

    def test_module_with_source(self, tmp_path):
        """Module with trackable source file should return hash."""
        mod_file = tmp_path / "my_module.py"
        mod_file.write_text("def hello(): return 42")
        mod = types.ModuleType("my_module")
        mod.__file__ = str(mod_file)
        tracker = MagicMock()
        tracker.tracked_modules = {"my_module"}
        tracker.dep_file_to_parents = {}
        assert self._component(tracker, mod, "my_module").startswith(":mod_src:")

    def test_value_none(self):
        assert self._component(MagicMock(), None, "x") == ""

    def test_hash_includes_dep_files(self, tmp_path):
        mod_file = tmp_path / "mod.py"
        mod_file.write_text("from helper import util")
        dep_file = tmp_path / "helper.py"
        dep_file.write_text("def util(): pass")
        mod = types.ModuleType("mod")
        mod.__file__ = str(mod_file)
        tracker = MagicMock()
        tracker.tracked_modules = {"mod"}
        tracker.dep_file_to_parents = {str(dep_file): {"mod"}}
        h1 = self._component(tracker, mod, "mod")
        dep_file.write_text("def util(): return 42")
        assert h1 != self._component(tracker, mod, "mod")

    def test_missing_module_file(self, tmp_path):
        mod = types.ModuleType("mod")
        mod.__file__ = str(tmp_path / "nofile.py")
        tracker = MagicMock()
        tracker.tracked_modules = {"mod"}
        tracker.dep_file_to_parents = {}
        assert self._component(tracker, mod, "mod") == ""

    def test_a_name_from_a_tracked_module_carries_its_source(self, tmp_path, monkeypatch):
        """`from helpers import clean` -- the case the simulator used to miss."""
        import sys

        mod_file = tmp_path / "zz_helpers_mod.py"
        mod_file.write_text("def clean(x): return x\nTHRESHOLD = 3\n")
        mod = types.ModuleType("zz_helpers_mod")
        mod.__file__ = str(mod_file)
        exec(mod_file.read_text(), mod.__dict__)
        monkeypatch.setitem(sys.modules, "zz_helpers_mod", mod)
        tracker = MagicMock()
        tracker.tracked_modules = {"zz_helpers_mod"}
        code = "from zz_helpers_mod import clean, THRESHOLD"
        # Narrowed to what each name reaches inside the module (`from_sym_src`)
        # since per-symbol keying; the whole-module `from_mod_src` is what a
        # name gets when its closure cannot be bounded. Either way it carries
        # source -- the thing this test is about.
        assert self._component(tracker, mod.clean, "clean", code).startswith(":from_sym_src:")
        assert self._component(tracker, 3, "THRESHOLD", code).startswith(":from_sym_src:")


# ===========================================================================
# UpstreamChecker initialization
# ===========================================================================


class TestUpstreamCheckerInit:
    """Test constructor and defaults."""

    def test_default_tracking_state(self):
        checker = _make_checker()
        assert isinstance(checker.variable_lineage, Mapping)
        assert isinstance(checker.executed_cell_codes, dict)

    def test_custom_tracking_state(self):
        state = TrackingState()
        state.lineage.record("test", "hash")
        checker = _make_checker(tracking_state=state)
        assert checker.variable_lineage["test"] == "hash"

    def test_simulation_cache_starts_empty(self):
        checker = _make_checker()
        assert len(checker.simulator.cache.entries) == 0

    def test_compute_hash_fn_stored(self):
        fn = lambda x: "custom"
        checker = _make_checker(compute_hash_fn=fn)
        assert checker.compute_hash_fn is fn
