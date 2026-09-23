"""A variable the upstream restore brings back is recorded like a statement hit.

Both restore paths hand back exactly a cache entry's value. The variable
restorer -- which brings one upstream variable back for a cell that needs it
-- recorded less about it than a statement hit does: not what the value was
built from, not its session hash, and it merged the entry's files into
whatever the name depended on before. With no record of its inputs, the
check for "built on an input that has been rebuilt since" passed for it, and
removing the definition of one of those inputs did not evict it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cash.backends import InMemoryBackend
from cash.notebook._protocols import TrackingState
from cash.notebook.restore import Restorer
from cash.tracking.file_dep_snapshot import snapshot_dependencies


def _restore(tmp_path, *, before: set[str] | None = None):
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    backend = InMemoryBackend()
    metadata = {
        "key": "entry-total",
        "code": "total = sum(values)",
        "source_hash": "source-total",
        "inputs": ["values"],
        "outputs": ["total"],
        "output_lineages": {"total": "lineage-total"},
        "input_lineages": {"values": "lineage-values"},
        "file_dependencies": snapshot_dependencies({str(data)}, set()),
        "execution_time": 1.0,
    }
    backend.set("entry-total", {"variables": {"total": 42}}, metadata)

    shell = MagicMock()
    shell.user_ns = {"values": [1, 2]}
    state = TrackingState()
    state.variable_sources["total"] = "entry-total"
    if before is not None:
        state.executed_file_deps["total"] = set(before)
    Restorer(shell, backend=backend, tracking_state=state).restore_variable("total")
    assert shell.user_ns["total"] == 42
    return state, str(data)


def test_it_records_what_it_was_built_from(tmp_path):
    state, _ = _restore(tmp_path)
    assert state.executed_input_lineages["total"] == {"values": "lineage-values"}


def test_it_records_its_session_hash(tmp_path):
    state, _ = _restore(tmp_path)
    assert state.current_session_hashes.get("total") in state.variable_hashes["total"]


def test_it_depends_on_exactly_its_entry_files(tmp_path):
    state, data = _restore(tmp_path, before={"/an/earlier/file.csv"})
    assert state.executed_file_deps["total"] == {data}
