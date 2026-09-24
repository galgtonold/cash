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
from cash.notebook._protocols import CashInstanceProtocol
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


def _reads_cash_instance(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "cash_instance") or (
        isinstance(node, ast.Attribute) and node.attr == "cash_instance"
    )


def test_every_member_the_notebook_reads_off_the_cash_instance_is_on_its_protocol():
    """The statement and upstream code hold the ``Cash`` object as a
    ``CashInstanceProtocol``, so what they read off it must be declared there.
    ``magics.py`` is left out: it holds the real ``Cash``."""
    used = set()
    for path in NOTEBOOK.rglob("*.py"):
        if path.name == "magics.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and _reads_cash_instance(node.value):
                used.add(node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and _reads_cash_instance(node.args[0])
                and isinstance(node.args[1], ast.Constant)
            ):
                used.add(node.args[1].value)

    assert "config" in used, "the scan no longer finds reads off the Cash instance"
    declared = set(CashInstanceProtocol.__annotations__) | {
        name for name in vars(CashInstanceProtocol) if not name.startswith("_")
    }
    assert used - declared == set()
