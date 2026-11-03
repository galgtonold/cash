"""The file writers above a cell share one read of the simulation trace.

Each writer's paths are resolved with the names the notebook binds to a
literal path (``OUT = Path('report')``), read from the trace of the
statements above. Every writer read the whole trace again for them, on
every cell.

Counted in reads of the trace, not timed.
"""

from __future__ import annotations

import os
import types

from cash.notebook.upstream import file_writers
from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.file_writers import FileWriterScheduler


def _entry(code: str, *outputs: str) -> TraceEntry:
    return TraceEntry(
        stmt_code=code, outputs=set(outputs), inputs=set(), input_hashes={}, produced_lineages={}, files_stale=False
    )


def _trace(n: int) -> list[TraceEntry]:
    trace = [_entry("OUT = 'report'", "OUT"), _entry("CHART = OUT / 'chart.png'", "CHART")]
    return trace + [_entry(f"x{i} = x{i - 1} + 1", f"x{i}") for i in range(1, n)]


def test_the_writers_above_a_cell_share_one_read_of_the_trace(monkeypatch):
    """Each writer above the cell asked for the bindings of the same trace:
    the trace was read once per writer, on every cell."""
    reads = 0
    real = file_writers._literal_path_bindings

    def counting(trace):
        nonlocal reads
        reads += 1
        return real(trace)

    monkeypatch.setattr(file_writers, "_literal_path_bindings", counting)
    probe = types.SimpleNamespace(record=lambda key: None)
    scheduler = FileWriterScheduler(types.SimpleNamespace(user_ns={}), types.SimpleNamespace(), probe)
    trace = _trace(10)
    first = scheduler._writer_paths("df.to_csv(OUT / 'a.csv')", trace)
    second = scheduler._writer_paths("fig.savefig(CHART)", trace)

    # Joined as this OS spells a path: report\a.csv on Windows.
    assert first == {os.path.join("report", "a.csv")} and second == {os.path.join("report", "chart.png")}
    assert reads == 1, f"the trace was read {reads} times for two writers"
    # A trace that grew is read again.
    trace.append(_entry("OUT = 'elsewhere'", "OUT"))
    scheduler._writer_paths("df.to_csv(OUT / 'a.csv')", trace)
    assert reads == 2
