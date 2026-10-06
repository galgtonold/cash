"""A value restored from a cache entry is recorded as the entry describes it.

Whichever restore brings it back, the value is exactly the entry's, so it is
recorded with what it was built from, its session hash, and exactly the
entry's files. With no record of its inputs, the check for "built on an
input that has been rebuilt since" passed for it.
"""

from __future__ import annotations

from cash.notebook.restored_var import apply_restored_var
from cash.notebook.statement import StatementCacheMetadata
from cash.notebook.tracking_state import TrackingState
from cash.tracking.file_dep_snapshot import snapshot_dependencies
from cash.value_hash import compute_hash


def _restore(tmp_path, *, before: set[str] | None = None):
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    metadata = StatementCacheMetadata.from_dict(
        {
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
    )
    state = TrackingState()
    if before is not None:
        state.executed_file_deps["total"] = set(before)
    apply_restored_var(state, "total", 42, metadata, compute_hash=compute_hash)
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
