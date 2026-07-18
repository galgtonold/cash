"""
Tests for provenance tracking module.
"""

import time
import json
import pytest

from cash.notebook.provenance import ProvenanceTracker, ProvenanceRecord


class TestProvenanceRecord:
    """Tests for the ProvenanceRecord dataclass."""

    def test_to_dict(self):
        record = ProvenanceRecord(
            variable="x",
            code="x = 10",
            inputs=[],
            timestamp=time.time(),
            status="computed",
            duration_ms=5.0,
            lineage_hash="abcdef1234567890abcdef",
        )
        d = record.to_dict()
        assert d["variable"] == "x"
        assert d["code"] == "x = 10"
        assert d["status"] == "computed"
        assert d["duration_ms"] == 5.0
        assert "..." in d["lineage_hash"]  # truncated

    def test_default_values(self):
        record = ProvenanceRecord(
            variable="y",
            code="y = 1",
            inputs=["x"],
            timestamp=0.0,
        )
        assert record.status == "computed"
        assert record.duration_ms == 0.0
        assert record.file_deps == []


class TestProvenanceTracker:
    """Tests for the ProvenanceTracker class."""

    def test_record_and_get_history(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 10", [], status="computed", duration_ms=1.0)
        history = tracker.get_history("x")
        assert len(history) == 1
        assert history[0].variable == "x"
        assert history[0].code == "x = 10"

    def test_get_latest(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 10", [])
        tracker.record("x", "x = 20", [])
        latest = tracker.get_latest("x")
        assert latest.code == "x = 20"

    def test_get_latest_missing(self):
        tracker = ProvenanceTracker()
        assert tracker.get_latest("nonexistent") is None

    def test_multiple_variables(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 10", [])
        tracker.record("y", "y = x * 2", ["x"])
        assert len(tracker.get_history("x")) == 1
        assert len(tracker.get_history("y")) == 1
        assert tracker.get_latest("y").inputs == ["x"]

    def test_get_dependencies(self):
        tracker = ProvenanceTracker()
        tracker.record("a", "a = 1", [])
        tracker.record("b", "b = a + 1", ["a"])
        tracker.record("c", "c = b * 2", ["b"])

        deps = tracker.get_dependencies("c")
        assert "b" in deps
        assert "a" in deps

    def test_get_dependents(self):
        tracker = ProvenanceTracker()
        tracker.record("a", "a = 1", [])
        tracker.record("b", "b = a + 1", ["a"])
        tracker.record("c", "c = a * 2", ["a"])

        dependents = tracker.get_dependents("a")
        assert "b" in dependents
        assert "c" in dependents

    def test_get_timeline(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 1", [])
        tracker.record("y", "y = 2", [])
        tracker.record("z", "z = 3", [])

        timeline = tracker.get_timeline()
        assert len(timeline) == 3
        assert timeline[0].variable == "x"
        assert timeline[2].variable == "z"

    def test_timeline_filter(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 1", [])
        tracker.record("y", "y = 2", [])
        tracker.record("x", "x = 3", [])

        timeline = tracker.get_timeline(variable="x")
        assert len(timeline) == 2
        assert all(r.variable == "x" for r in timeline)

    def test_max_history_per_var(self):
        tracker = ProvenanceTracker()
        tracker.max_history_per_var = 5
        for i in range(10):
            tracker.record("x", f"x = {i}", [])
        assert len(tracker.get_history("x")) == 5

    def test_max_timeline(self):
        tracker = ProvenanceTracker()
        tracker.max_timeline = 10
        for i in range(20):
            tracker.record(f"v{i}", f"v{i} = {i}", [])
        assert len(tracker.get_timeline(limit=100)) == 10

    def test_format_provenance(self):
        tracker = ProvenanceTracker()
        tracker.record("result", "result = compute(data)", ["data"],
                       status="computed", duration_ms=150.0)
        output = tracker.format_provenance("result")
        assert "result" in output
        assert "computed" in output
        assert "150.0ms" in output
        assert "data" in output

    def test_format_provenance_missing(self):
        tracker = ProvenanceTracker()
        output = tracker.format_provenance("nonexistent")
        assert "No provenance" in output

    def test_format_with_graph(self):
        tracker = ProvenanceTracker()
        tracker.record("a", "a = 1", [])
        tracker.record("b", "b = a + 1", ["a"])
        output = tracker.format_provenance("b", show_graph=True)
        assert "Dependency Graph" in output
        assert "a" in output

    def test_format_with_timeline(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 1", [], status="computed")
        tracker.record("x", "x = 2", [], status="restored")
        output = tracker.format_provenance("x", show_timeline=True)
        assert "Timeline" in output
        assert "🔧" in output or "📦" in output

    def test_to_json(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 1", [], lineage_hash="abc123")
        result = json.loads(tracker.to_json("x"))
        assert len(result) == 1
        assert result[0]["variable"] == "x"

    def test_to_json_all(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 1", [])
        tracker.record("y", "y = 2", [])
        result = json.loads(tracker.to_json())
        assert len(result) == 2

    def test_clear(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 1", [])
        tracker.clear()
        assert len(tracker.tracked_variables) == 0
        assert len(tracker.get_timeline()) == 0

    def test_tracked_variables(self):
        tracker = ProvenanceTracker()
        tracker.record("x", "x = 1", [])
        tracker.record("y", "y = 2", ["x"])
        assert tracker.tracked_variables == {"x", "y"}

    def test_circular_dependency_safe(self):
        """get_dependencies should not infinite loop on circular refs."""
        tracker = ProvenanceTracker()
        tracker.record("a", "a = b + 1", ["b"])
        tracker.record("b", "b = a + 1", ["a"])
        # This should not hang
        deps = tracker.get_dependencies("a")
        assert "b" in deps

    def test_file_deps_tracked(self):
        tracker = ProvenanceTracker()
        tracker.record("df", "df = pd.read_csv('data.csv')", [],
                       file_deps=["data.csv"])
        latest = tracker.get_latest("df")
        assert "data.csv" in latest.file_deps


class TestCashProvenanceMagic:
    """Tests for the %cash_provenance magic command."""

    @pytest.fixture
    def magics_fixture(self):
        from cash.notebook.ipython.magics import CashMagics
        from cash.core import Cash
        from cash.backends import InMemoryBackend
        from traitlets.config.configurable import Configurable
        from unittest.mock import MagicMock

        class MockShell(Configurable):
            def __init__(self):
                super().__init__()
                self.user_ns = {}
                self.input_transformers_cleanup = []
                self.run_cell = MagicMock()
                self.events = MagicMock()
                self.ast_transformers = []
                self.user_global_ns = self.user_ns

        backend = InMemoryBackend()
        cash = Cash(backend=backend, register_magic=False)
        shell = MockShell()
        magics = CashMagics(shell, cash)
        magics._auto_cache_enabled = True
        yield magics, shell, backend
        backend.clear()

    def test_list_empty(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics.cash_provenance("--all")
        output = capsys.readouterr().out
        assert "No provenance" in output

    def test_list_with_data(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics._session.provenance.record("x", "x = 1", [], status="computed")
        magics.cash_provenance("--all")
        output = capsys.readouterr().out
        assert "x" in output
        assert "1 records" in output

    def test_show_variable(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics._session.provenance.record("result", "result = calc()", ["data"],
                                  status="computed", duration_ms=50.0)
        magics.cash_provenance("result")
        output = capsys.readouterr().out
        assert "result" in output
        assert "computed" in output

    def test_clear(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics._session.provenance.record("x", "x = 1", [])
        magics.cash_provenance("--clear")
        output = capsys.readouterr().out
        assert "cleared" in output
        assert len(magics._session.provenance.tracked_variables) == 0

    def test_json_output(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics._session.provenance.record("x", "x = 1", [])
        magics.cash_provenance("x --json")
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["variable"] == "x"


class TestFileDepsDisplayHygiene:
    """Round-8 gate (P2): %cash_provenance dumped 100+ phantom venv
    entry_points.txt paths into "File deps", so a variable looked like it
    depended on all of site-packages. Those are importlib.metadata probes for
    files that never existed — already ignored for freshness (CAS-185), but they
    leaked into the display."""

    def _tracker(self):
        from cash.notebook.provenance import ProvenanceTracker
        return ProvenanceTracker()

    def test_phantom_paths_are_not_displayed(self, tmp_path):
        real = tmp_path / "sales.csv"
        real.write_text("a,b\n")
        phantoms = [str(tmp_path / "venv" / f"pkg{i}" / "entry_points.txt")
                    for i in range(120)]

        t = self._tracker()
        t.record(variable="df", code="df = load()", inputs=[], status="COMPUTED",
                 file_deps=[str(real), *phantoms])
        out = t.format_provenance("df")

        assert "entry_points.txt" not in out
        assert "sales.csv" in out

    def test_long_real_dep_list_is_capped_with_a_count(self, tmp_path):
        paths = []
        for i in range(25):
            p = tmp_path / f"f{i}.csv"
            p.write_text("x")
            paths.append(str(p))

        t = self._tracker()
        t.record(variable="v", code="v = 1", inputs=[], status="COMPUTED",
                 file_deps=paths)
        out = t.format_provenance("v")

        assert "(+17 more)" in out          # 25 real - 8 shown
        assert out.count("f0.csv") == 1

    def test_all_phantom_deps_render_no_file_deps_line(self, tmp_path):
        t = self._tracker()
        t.record(variable="x", code="x = 1", inputs=[], status="COMPUTED",
                 file_deps=[str(tmp_path / "nope" / "entry_points.txt")])
        assert "File deps" not in t.format_provenance("x")
