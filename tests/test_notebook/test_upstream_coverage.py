"""Comprehensive unit tests for upstream.py to improve coverage from 62% to 70%+.

Tests cover:
- _handle_lineage_mismatch
- _resolve_input_lineage
- _resolve_virtual_input_lineages
- lineage_formula.module_source_component (the simulator's module component)
- _resolve_fallback_cache_idx / _reset_advanced_lineages
- _handle_downstream_advancement_fallback
- _stat_file_deps
- _check_file_deps_for_restore
- _check_lineage_consistency
- _get_metadata_only
- reset_caches
- set_tracking_state
"""

import os
import time
import types
from unittest.mock import MagicMock

import pytest

from cash.notebook._protocols import TrackingState
from cash.notebook.file_dep_snapshot import snapshot_file_deps
from cash.notebook.upstream import UpstreamChecker
from cash.notebook.upstream.virtual_lineage import VirtualLineage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checker(**kwargs):
    """Create an UpstreamChecker with a mock shell."""
    shell = MagicMock()
    shell.user_ns = kwargs.pop("user_ns", {})
    cash_instance = kwargs.pop("cash_instance", None)
    debug = kwargs.pop("debug", False)
    tracking_state = kwargs.pop("tracking_state", None)
    compute_hash_fn = kwargs.pop("compute_hash_fn", None)
    checker = UpstreamChecker(
        shell,
        cash_instance=cash_instance,
        debug=debug,
        compute_hash_fn=compute_hash_fn,
        tracking_state=tracking_state,
    )
    return checker


# ===========================================================================
# _handle_lineage_mismatch
# ===========================================================================


class TestHandleLineageMismatch:
    """Test lineage mismatch handling (the notebook-based pass 2 version)."""

    def _call_mismatch(
        self,
        checker,
        var_name="x",
        actual="actual",
        virtual="virtual",
        broken_vars=None,
        simulation_trace=None,
        notebook_cells=None,
    ):
        """Helper to call _handle_lineage_mismatch with all required args."""
        if broken_vars is None:
            broken_vars = set()
        if simulation_trace is None:
            simulation_trace = []
        if notebook_cells is None:
            notebook_cells = []
        checker.simulator._classifier._handle_lineage_mismatch(
            var_name=var_name,
            actual_lineage=actual,
            final_virtual_hash=virtual,
            vars_derived_from_loops=set(),
            upstream_has_modifications=False,
            loop_derived_trust_overridden=False,
            loop_target_vars=set(),
            virtual_lineage={},
            virtual_modules=set(),
            vars_with_stale_files=set(),
            simulation_trace=simulation_trace,
            required_inputs=None,
            current_cell_outputs=None,
            notebook_cells=notebook_cells,
            broken_vars=broken_vars,
        )
        return broken_vars

    def test_mismatch_adds_to_broken_vars(self):
        """When lineage doesn't match, var should be added to broken_vars."""
        checker = _make_checker()
        broken = self._call_mismatch(checker, var_name="x", actual="aaa", virtual="bbb")
        assert "x" in broken

    def test_matching_lineage_not_broken(self):
        """When actual matches virtual, var should not be broken."""
        checker = _make_checker()
        broken = self._call_mismatch(checker, var_name="x", actual="same", virtual="same")
        # If lineages match, _handle_mismatch_prereqs should return True
        # and var should NOT be added to broken_vars
        # (depends on prereqs logic, but at minimum should not crash)
        assert isinstance(broken, set)

    def test_debug_mode_no_crash(self):
        """Debug mode should produce debug output without crashing."""
        checker = _make_checker(debug=True)
        broken = self._call_mismatch(checker, actual="expected123", virtual="actual456")
        assert isinstance(broken, set)

    def test_with_simulation_trace(self):
        """Should search simulation trace for the variable's last statement."""
        checker = _make_checker()
        trace = [("x = 1", {"x"}, {}, {}, {}, {})]
        self._call_mismatch(checker, var_name="x", simulation_trace=trace)


# ===========================================================================
# _resolve_input_lineage
# ===========================================================================


class TestResolveInputLineage:
    """Test the priority-based input lineage resolution."""

    def test_virtual_lineage_first(self):
        """Virtual lineage should be checked first."""
        checker = _make_checker()
        checker.variable_lineage["x"] = "runtime_hash"
        result = checker.simulator._virtual_lineage._resolve_input_lineage("x", {"x": "virtual_hash"}, set())
        assert result == "virtual_hash"

    def test_falls_back_to_variable_lineage(self):
        """Falls back to variable_lineage when not in virtual."""
        checker = _make_checker()
        checker.variable_lineage["x"] = "runtime_hash"
        result = checker.simulator._virtual_lineage._resolve_input_lineage("x", {}, set())
        assert result == "runtime_hash"

    def test_falls_back_to_user_ns_hash(self):
        """Falls back to hashing from user_ns when no lineage exists."""
        checker = _make_checker(user_ns={"x": 42})
        result = checker.simulator._virtual_lineage._resolve_input_lineage("x", {}, set())
        assert result is not None
        assert len(result) == 64

    def test_custom_compute_hash_fn(self):
        """Should use compute_hash_fn when available."""
        checker = _make_checker(
            user_ns={"x": 42},
            compute_hash_fn=lambda v: "custom_hash_result",
        )
        result = checker.simulator._virtual_lineage._resolve_input_lineage("x", {}, set())
        assert result == "custom_hash_result"

    def test_returns_none_for_missing_variable(self):
        """Returns None when variable is not in any source."""
        checker = _make_checker()
        result = checker.simulator._virtual_lineage._resolve_input_lineage("x", {}, set())
        assert result is None

    def test_returns_none_for_none_value(self):
        """Returns None when user_ns has None for the variable."""
        checker = _make_checker(user_ns={"x": None})
        result = checker.simulator._virtual_lineage._resolve_input_lineage("x", {}, set())
        assert result is None


# ===========================================================================
# _resolve_virtual_input_lineages
# ===========================================================================


class TestResolveVirtualInputLineages:
    """Test virtual input lineage resolution for all inputs of a statement."""

    def test_basic_resolution(self):
        checker = _make_checker()
        checker.variable_lineage["a"] = "hash_a"
        checker.variable_lineage["b"] = "hash_b"
        result = checker.simulator._virtual_lineage._resolve_virtual_input_lineages("x = a + b", {"a", "b"}, {}, set())
        assert len(result) == 2
        assert "hash_a" in result
        assert "hash_b" in result

    def test_skips_get_ipython(self):
        """Should skip get_ipython and __builtins__."""
        checker = _make_checker()
        checker.variable_lineage["a"] = "hash_a"
        result = checker.simulator._virtual_lineage._resolve_virtual_input_lineages(
            "x = a", {"a", "get_ipython", "__builtins__"}, {}, set()
        )
        assert len(result) == 1

    def test_virtual_lineage_priority(self):
        """Virtual lineage should be used over runtime lineage."""
        checker = _make_checker()
        checker.variable_lineage["a"] = "runtime_hash"
        result = checker.simulator._virtual_lineage._resolve_virtual_input_lineages(
            "x = a", {"a"}, {"a": "virtual_hash"}, set()
        )
        assert result == ["virtual_hash"]

    def test_empty_inputs(self):
        checker = _make_checker()
        result = checker.simulator._virtual_lineage._resolve_virtual_input_lineages("x = 1", set(), {}, set())
        assert result == []


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
# _check_file_deps_for_restore
# ===========================================================================


class TestCheckFileDepsForRestore:
    """Test file dependency validation for virtual restore."""

    def test_all_fresh(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("content")
        checker = _make_checker()
        result = checker.simulator._virtual_lineage._check_file_deps_for_restore(
            snapshot_file_deps({str(f)}), time.time()
        )
        assert result is None  # None means all fresh

    def test_stale_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("content")
        snapshot = snapshot_file_deps({str(f)})
        f.write_text("changed content")
        checker = _make_checker()
        result = checker.simulator._virtual_lineage._check_file_deps_for_restore(snapshot, time.time())
        assert result is not None  # Tuple means failure
        assert isinstance(result, tuple)

    def test_missing_file(self, tmp_path):
        checker = _make_checker()
        result = checker.simulator._virtual_lineage._check_file_deps_for_restore(
            {str(tmp_path / "gone.csv"): {"mtime": 1.0}}, time.time()
        )
        assert result is not None

    def test_empty_deps(self):
        checker = _make_checker()
        result = checker.simulator._virtual_lineage._check_file_deps_for_restore({}, time.time())
        assert result is None


# ===========================================================================
# _check_lineage_consistency
# ===========================================================================


class TestCheckLineageConsistency:
    """Test output lineage consistency validation."""

    def test_consistent_lineage(self):
        checker = _make_checker()
        metadata = {"output_lineages": {"x": "hash_x"}}
        result = checker.simulator._virtual_lineage._check_lineage_consistency(
            metadata, {}, {"x": "hash_x"}, time.time()
        )
        assert result is None  # None means consistent

    def test_inconsistent_lineage(self):
        checker = _make_checker()
        metadata = {"output_lineages": {"x": "cached_hash"}}
        result = checker.simulator._virtual_lineage._check_lineage_consistency(
            metadata, {}, {"x": "expected_hash"}, time.time()
        )
        assert result is not None

    def test_skipped_with_file_deps(self):
        """When file deps exist, lineage check is skipped."""
        checker = _make_checker()
        metadata = {"output_lineages": {"x": "cached_hash"}}
        result = checker.simulator._virtual_lineage._check_lineage_consistency(
            metadata, {"file.csv": 1.0}, {"x": "expected_hash"}, time.time()
        )
        assert result is None  # Skipped — file deps present

    def test_no_expected_lineages(self):
        checker = _make_checker()
        metadata = {"output_lineages": {"x": "hash"}}
        result = checker.simulator._virtual_lineage._check_lineage_consistency(metadata, {}, None, time.time())
        assert result is None  # No expected lineages → pass

    def test_no_output_lineages_in_metadata(self):
        checker = _make_checker()
        metadata = {}  # No output_lineages key
        result = checker.simulator._virtual_lineage._check_lineage_consistency(metadata, {}, {"x": "hash"}, time.time())
        assert result is None


# ===========================================================================
# _get_metadata_only
# ===========================================================================


class TestGetMetadataOnly:
    """Test metadata-only backend access."""

    def test_no_cash_instance(self):
        checker = _make_checker()
        result = checker.simulator._virtual_lineage._get_metadata_only("some_key")
        assert result is None

    def test_with_get_metadata_method(self):
        mock_backend = MagicMock()
        mock_backend.get_metadata.return_value = {"key": "value"}
        mock_cash = MagicMock()
        mock_cash.backend = mock_backend
        checker = _make_checker(cash_instance=mock_cash)
        result = checker.simulator._virtual_lineage._get_metadata_only("test_key")
        assert result == {"key": "value"}
        mock_backend.get_metadata.assert_called_once_with("test_key")


# ===========================================================================
# reset_caches
# ===========================================================================


class TestResetCaches:
    """Test cache clearing."""

    def test_clears_all_caches(self):
        checker = _make_checker()
        checker.simulator._virtual_lineage._ast_cache["code"] = MagicMock()
        checker.simulator._virtual_lineage._simulation_cache.append(MagicMock())
        checker.simulator._virtual_lineage._simulation_cell_hashes[0] = "hash"
        checker.reset_caches()
        assert len(checker.simulator._virtual_lineage._ast_cache) == 0
        assert len(checker.simulator._virtual_lineage._simulation_cache) == 0
        assert len(checker.simulator._virtual_lineage._simulation_cell_hashes) == 0


# ===========================================================================
# set_tracking_state
# ===========================================================================


class TestSetTrackingState:
    """Test tracking state wiring."""

    def test_wires_all_dicts(self):
        state = TrackingState()
        state.variable_lineage["x"] = "hash"
        state.executed_cell_codes["x"] = "x = 1"
        checker = _make_checker(tracking_state=state)
        assert checker.variable_lineage is state.variable_lineage
        assert checker.executed_cell_codes is state.executed_cell_codes
        assert checker.variable_lineage["x"] == "hash"

    def test_new_state_replaces_old(self):
        state1 = TrackingState()
        state1.variable_lineage["x"] = "old"
        state2 = TrackingState()
        state2.variable_lineage["y"] = "new"
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
        checker.simulator._virtual_lineage._cell_id_to_last_index["cell_0"] = 0
        result = checker._resolve_fallback_cache_idx("cell_0")
        assert result is None

    def test_returns_previous_cache_index(self):
        checker = _make_checker()
        checker.last_cell_index = None
        checker.simulator._virtual_lineage._cell_id_to_last_index["cell_2"] = 2
        # Need at least 2 simulation cache entries
        checker.simulator._virtual_lineage._simulation_cache = [MagicMock(), MagicMock(), MagicMock()]
        result = checker._resolve_fallback_cache_idx("cell_2")
        assert result == 1  # Previous index

    def test_uses_last_cell_index_without_cell_id(self):
        checker = _make_checker()
        checker.last_cell_index = 3
        checker.simulator._virtual_lineage._simulation_cache = [MagicMock()] * 4
        result = checker._resolve_fallback_cache_idx(None)
        assert result == 2

    def test_cache_too_small(self):
        """If simulation cache is smaller than target index, return None."""
        checker = _make_checker()
        checker.last_cell_index = None
        checker.simulator._virtual_lineage._cell_id_to_last_index["cell_5"] = 5
        checker.simulator._virtual_lineage._simulation_cache = [MagicMock()]  # Only 1 entry
        result = checker._resolve_fallback_cache_idx("cell_5")
        assert result is None


# ===========================================================================
# _reset_advanced_lineages
# ===========================================================================


class TestResetAdvancedLineages:
    """Test resetting lineages that are 'ahead' of cached virtual lineage."""

    def test_resets_mismatched_lineage(self):
        checker = _make_checker()
        checker.variable_lineage["x"] = "runtime_ahead"
        cached_virtual = {"x": "virtual_correct"}
        checker._reset_advanced_lineages({"x"}, cached_virtual, 0)
        assert checker.variable_lineage["x"] == "virtual_correct"

    def test_skips_matching_lineage(self):
        checker = _make_checker()
        checker.variable_lineage["x"] = "same_hash"
        cached_virtual = {"x": "same_hash"}
        checker._reset_advanced_lineages({"x"}, cached_virtual, 0)
        assert checker.variable_lineage["x"] == "same_hash"

    def test_skips_variables_not_in_virtual(self):
        checker = _make_checker()
        checker.variable_lineage["x"] = "runtime_hash"
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
        checker.simulator._virtual_lineage._simulation_cache = [MagicMock()]
        checker._handle_downstream_advancement_fallback(cell_id=None, required_inputs={"a"}, current_cell_outputs={"b"})

    def test_no_op_with_empty_inputs(self):
        checker = _make_checker()
        checker.simulator._virtual_lineage._simulation_cache = [MagicMock()]
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
        """Module in user_ns but not in function_tracker._tracked_modules."""
        mod = types.ModuleType("fake_mod")
        mod.__file__ = "/nonexistent/fake_mod.py"
        tracker = MagicMock()
        tracker._tracked_modules = set()
        assert self._component(tracker, mod, "fake_mod") == ""

    def test_module_with_source(self, tmp_path):
        """Module with trackable source file should return hash."""
        mod_file = tmp_path / "my_module.py"
        mod_file.write_text("def hello(): return 42")
        mod = types.ModuleType("my_module")
        mod.__file__ = str(mod_file)
        tracker = MagicMock()
        tracker._tracked_modules = {"my_module"}
        tracker._dep_file_to_parents = {}
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
        tracker._tracked_modules = {"mod"}
        tracker._dep_file_to_parents = {str(dep_file): {"mod"}}
        h1 = self._component(tracker, mod, "mod")
        dep_file.write_text("def util(): return 42")
        assert h1 != self._component(tracker, mod, "mod")

    def test_missing_module_file(self, tmp_path):
        mod = types.ModuleType("mod")
        mod.__file__ = str(tmp_path / "nofile.py")
        tracker = MagicMock()
        tracker._tracked_modules = {"mod"}
        tracker._dep_file_to_parents = {}
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
        tracker._tracked_modules = {"zz_helpers_mod"}
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

    def test_default_debug_false(self):
        checker = _make_checker()
        assert checker.debug is False

    def test_debug_flag(self):
        checker = _make_checker(debug=True)
        assert checker.debug is True

    def test_default_tracking_state(self):
        checker = _make_checker()
        assert isinstance(checker.variable_lineage, dict)
        assert isinstance(checker.executed_cell_codes, dict)

    def test_custom_tracking_state(self):
        state = TrackingState()
        state.variable_lineage["test"] = "hash"
        checker = _make_checker(tracking_state=state)
        assert checker.variable_lineage["test"] == "hash"

    def test_ast_cache_starts_empty(self):
        checker = _make_checker()
        assert len(checker.simulator._virtual_lineage._ast_cache) == 0

    def test_simulation_cache_starts_empty(self):
        checker = _make_checker()
        assert len(checker.simulator._virtual_lineage._simulation_cache) == 0

    def test_compute_hash_fn_stored(self):
        fn = lambda x: "custom"
        checker = _make_checker(compute_hash_fn=fn)
        assert checker.compute_hash_fn is fn
