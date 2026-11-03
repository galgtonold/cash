"""The names a notebook binds to a literal path are read once per statement.

To tell what a file writer above the cell writes after a restart, the
upstream check reads the path each name holds from the statements above
(``OUT = Path('report')``). It parsed every statement above again for each
writer, on every cell: in a 174-cell notebook with a dozen ``save_fig``
calls, 310,000 parses and tens of seconds of a first Run All.

Counted in parses, not timed.
"""

from __future__ import annotations

import ast
import os
from unittest.mock import patch

from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.file_writers import _literal_path_bindings


def _entry(code: str, *outputs: str) -> TraceEntry:
    return TraceEntry(
        stmt_code=code, outputs=set(outputs), inputs=set(), input_hashes={}, produced_lineages={}, files_stale=False
    )


def _trace(n: int) -> list[TraceEntry]:
    trace = [_entry("OUT = 'report'", "OUT"), _entry("CHART = OUT / 'chart.png'", "CHART")]
    trace += [_entry(f"x{i} = x{i - 1} + 1", f"x{i}") for i in range(1, n)]
    trace.append(_entry("rows.append(1)", "rows"))
    return trace


def _parses(trace: list[TraceEntry]) -> tuple[dict, int]:
    real = ast.parse
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    with patch.object(ast, "parse", counting):
        found = _literal_path_bindings(trace)
    return found, calls


def test_a_second_writer_reads_the_statements_above_without_parsing_them():
    # What it finds is as before: a name bound twice to different paths, or
    # bound by anything but one literal path, is left out; a call that only
    # changes a name in place does not bind it.
    trace = _trace(40) + [
        _entry("TWICE = 'a'", "TWICE"),
        _entry("TWICE = 'b'", "TWICE"),
        _entry("KEEP = 'k'", "KEEP"),
        _entry("KEEP.strip()", "KEEP"),
        _entry("SAME = 's'", "SAME"),
        _entry("SAME = 's'", "SAME"),
        _entry("DERIVED = TWICE / 'x'", "DERIVED"),
    ]
    first, _ = _parses(trace)
    again, parses = _parses(list(trace))

    chart = os.path.join("report", "chart.png")  # report\\chart.png on Windows
    assert first == again == {"OUT": "report", "CHART": chart, "KEEP": "k", "SAME": "s"}
    assert parses == 0, f"{parses} parses for statements already read"
