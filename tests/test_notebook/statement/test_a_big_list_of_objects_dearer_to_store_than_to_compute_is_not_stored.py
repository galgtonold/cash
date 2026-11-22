"""A statement building a big list of Python objects that costs more to
store than a hit of it saves is not stored, and runs again.

1.5 million ``SimpleNamespace`` records built in 1.5 s were priced by
their bytes (0.07 s to restore), then walked by the share check until its
time ran out; let through, the RAM tier copied them for 8 s, an object at a
time, and a hit copied them again. The store is priced before the walks by
a sample of the records (`many_objects_cost`) and refused, with the reason
said; the share and in-place checks, which only decide whether the value
can be stored, do not walk it.

The sample is measured as the store runs; its price is pinned here
(`priced_at`) and the compute too (`computed_in`), so a loaded machine
cannot move the verdict. That the sample finds the records is checked.
"""

from __future__ import annotations

import pytest

from cash.backends import sampled_cost
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import in_place, processor, store
from cash.notebook.statement.processor import StatementProcessor
from cash.notebook.statement.rebuild_cost import RebuildCostLedger
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

BUILD = (
    f"records = time.sleep({ABOVE_PERSISTENCE_FLOOR_S}) or "
    "[types.SimpleNamespace(i=i, name=f'n{i}') for i in range(30_000)]"
)
READ = "total = sum(r.i for r in records)"


@pytest.fixture(autouse=True)
def share_check_to_spare(monkeypatch):
    """The control's share check of 30,000 records gets the floor next to
    a short statement; past it the statement only re-runs."""
    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 5.0)


@pytest.fixture
def priced_at(monkeypatch):
    """Pin what storing the records costs, copying and pickling alike;
    loading them takes what pickling does. Returns the counts of records
    the real sample found."""
    found: list[int] = []

    def pin(seconds: float) -> list[int]:
        def many_objects_cost(value):
            measured = sampled_cost.many_objects_cost(value)
            if measured is None:
                return None
            found.append(measured.items)
            return measured._replace(copy=seconds, dump=seconds, load=seconds)

        monkeypatch.setattr(store, "many_objects_cost", many_objects_cost)
        return found

    return pin


@pytest.fixture
def computed_in(monkeypatch):
    monkeypatch.setattr(RebuildCostLedger, "final_rebuild_seconds", lambda *_args, **_kwargs: 1.0)


@pytest.fixture
def walked(monkeypatch) -> list[str]:
    """The in-place and share checks of a statement's outputs, as they run."""
    calls: list[str] = []
    refusal, share = in_place.refusal, StatementProcessor._refuse_unrestorable_outputs

    def counted_refusal(*args, **kwargs):
        calls.append("in_place")
        return refusal(*args, **kwargs)

    def counted_share(self, *args, **kwargs):
        calls.append("share")
        return share(self, *args, **kwargs)

    monkeypatch.setattr(in_place, "refusal", counted_refusal)
    monkeypatch.setattr(StatementProcessor, "_refuse_unrestorable_outputs", counted_share)
    return calls


def _row(cash_magics, start: str) -> dict:
    rows = [s for s in cash_magics._last_cell_metrics["statements"] if s["code"].startswith(start)]
    assert len(rows) == 1, cash_magics._last_cell_metrics["statements"]
    return rows[0]


def _build_twice(cash_magics, walked: list[str]) -> tuple[dict, dict, list[str]]:
    """The build's rows, both times, and what walked it the first time."""
    run_cash_cell(cash_magics, "import time, types")
    rows = []
    for _ in range(2):
        walked.clear()
        run_cash_cell(cash_magics, BUILD)
        rows.append(_row(cash_magics, "records ="))
        if len(rows) == 1:
            first_walks = list(walked)
        run_cash_cell(cash_magics, READ)
        assert cash_magics.shell.user_ns["total"] == sum(range(30_000))
    return rows[0], rows[1], first_walks


def test_records_dearer_to_store_than_to_build_are_not_stored(cash_magics, priced_at, computed_in, walked):
    found = priced_at(60.0)
    first, again, walks = _build_twice(cash_magics, walked)

    assert found and set(found) == {30_000}
    assert "30,000 objects would take" in (first.get("skipped_reason") or ""), first
    assert walks == [], walks
    assert again["status"] == CacheStatus.COMPUTED


def test_records_cheap_to_store_are_stored(cash_magics, priced_at, computed_in, walked):
    """The control: the same records, priced cheap, are share-checked and
    stored, and the second run restores them."""
    found = priced_at(0.001)
    first, again, walks = _build_twice(cash_magics, walked)

    assert found and set(found) == {30_000}
    assert not first.get("skipped_reason"), first
    assert "share" in walks, walks
    assert again["status"] == CacheStatus.RESTORED
