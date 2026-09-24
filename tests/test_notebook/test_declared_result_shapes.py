"""The TypedDicts that describe a statement's result must name every key the
code writes.

No type checker runs over ``cash.notebook``, so nothing else notices when a
key is added to a result, a call event or the cell's timing breakdown and
not to its declaration. These tests run cells through the pipeline and
compare what came out with what is declared.
"""

from __future__ import annotations

import ast
from pathlib import Path

from cash.backends import CacheBackend
from cash.notebook.ipython._types import TimingBreakdown
from cash.notebook.ipython.cell_executor import PipelineCompleted
from cash.notebook.statement.results import DecoratorCallMetric, ProcessResult
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

NOTEBOOK = Path(__file__).resolve().parents[2] / "src" / "cash" / "notebook"

CELLS = [
    f"import time\ndef slow(k):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return k * 2",
    "a = slow(3)",
    "a = slow(3)",
    "acc = []\nfor k in range(3):\n    acc.append(slow(k))",
    "if a > 1:\n    b = a + 1",
    "import random\nr = random.random()",
    "print(a)",
]


def _declared(typed_dict) -> set[str]:
    return set(typed_dict.__required_keys__) | set(typed_dict.__optional_keys__)


def _run_cells(magics) -> list[PipelineCompleted]:
    runs = []
    for code in CELLS:
        done = magics._cell_executor.execute_cell(code, cell_id=magics.resolve_cell_id())
        assert isinstance(done, PipelineCompleted), code
        magics._finalize_cell_body(code, done)
        runs.append(done)
    return runs


def test_every_key_a_result_carries_is_declared(cash_magics):
    runs = _run_cells(cash_magics)
    result_keys = {key for done in runs for m in done.all_metrics for key in m}
    event_keys = {key for done in runs for m in done.all_metrics for e in m.get("decorator_calls") or () for key in e}
    timing_keys = {key for done in runs for key in done.timing_breakdown}

    assert event_keys, "no intercepted call was recorded: the cells no longer exercise call events"
    assert result_keys - _declared(ProcessResult) == set()
    assert event_keys - _declared(DecoratorCallMetric) == set()
    assert timing_keys - _declared(TimingBreakdown) == set()


def test_every_key_a_decorated_call_reports_is_declared(cash_instance):
    @cash_instance.cache
    def double(k):
        return k * 2

    double(1)
    double(1)
    events = cash_instance.drain_decorator_calls()

    assert [e["cache_hit"] for e in events] == [False, True]
    assert {key for e in events for key in e} - _declared(DecoratorCallMetric) == set()


def test_every_backend_method_the_notebook_calls_is_on_the_backend_base():
    """``CashInstanceProtocol.backend`` is typed as :class:`CacheBackend`, so a
    method the notebook calls on a backend must be declared there."""
    used = set()
    for path in NOTEBOOK.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, (ast.Name, ast.Attribute))
                and (node.value.id if isinstance(node.value, ast.Name) else node.value.attr) in ("backend", "_backend")
            ):
                used.add(node.attr)

    assert "set_metadata_only" in used, "the scan no longer finds backend calls"
    assert {name for name in used if not hasattr(CacheBackend, name)} == set()
