"""A statement whose value costs more to store than a hit of it saves is
not stored, and runs again.

A loop building 6,000 small frames in 1.38 s had them copied into the RAM
tier (0.55 s) and pickled for disk (0.36 s, in the background, holding the
interpreter lock while the next statement ran, which took 1.1 s for its
0.34 s); and a hit from RAM copies every table again. The store is priced
before anything is copied, by a sample of the tables (`many_frames_cost`),
and refused when copying plus twice the pickling costs more than the
compute less the restore, with the reason said. It is priced before the
walks that only decide whether the value can be stored, which a refused
store leaves out.

The sample is measured as the store runs; its price is pinned here
(`priced_at`), and the compute too where the verdict turns on it
(`computed_in`), so a loaded machine, which slows the statement and the
sample unevenly, cannot move the verdict. That the sample finds the
tables is checked on every call.
"""

from __future__ import annotations

import ast

import pytest

pytest.importorskip("pandas")

from cash.analysis.annotations import get_statement_annotations
from cash.backends import FileBackend, InMemoryBackend, TieredBackend, sampled_cost
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import in_place, store
from cash.notebook.statement.processor import StatementProcessor
from cash.notebook.statement.rebuild_cost import RebuildCostLedger
from tests._cell_driver import run_cash_cell

SETUP = "import time\nimport pandas as pd\nt = pd.DataFrame({'user': ['s10'] * 3, 'n': [1, 2, 3]})"
BUILD = "frames = [t.copy() for _ in range(1000)]"


@pytest.fixture
def priced_at(monkeypatch):
    """Pin what storing the tables costs: *seconds* split evenly between
    copying and pickling, or *copy* and *dump* apart; loading them takes
    what pickling does. Returns the counts of tables the real sample found."""
    found: list[int] = []

    def pin(seconds: float = 0.0, *, copy: float | None = None, dump: float | None = None) -> list[int]:
        copy_s = seconds / 2 if copy is None else copy
        dump_s = seconds / 2 if dump is None else dump

        def many_frames_cost(value):
            measured = sampled_cost.many_frames_cost(value)
            if measured is None:
                return None
            found.append(measured.items)
            return measured._replace(copy=copy_s, dump=dump_s, load=dump_s)

        monkeypatch.setattr(store, "many_frames_cost", many_frames_cost)
        return found

    return pin


@pytest.fixture
def computed_in(monkeypatch):
    """Pin what getting the value back without its entry costs (the
    compute the store is weighed against) to *seconds*."""

    def pin(seconds: float) -> None:
        monkeypatch.setattr(RebuildCostLedger, "final_rebuild_seconds", lambda *_args, **_kwargs: seconds)

    return pin


@pytest.fixture
def on_disk(cash_instance, tmp_path):
    """The default tiers, RAM then disk: a store copies the tables and
    pickles them, and a hit copies them again from RAM."""
    disk = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    cash_instance.backend = TieredBackend([InMemoryBackend(), disk])
    yield
    disk._writes.wait_all()
    disk.shutdown()


def _twice(cash_magics, code):
    """What running *code* as a statement says, both times."""
    run_cash_cell(cash_magics, SETUP)
    node = ast.parse(code).body[0]
    results = []
    for _ in range(2):
        cash_magics.shell.user_ns.pop("frames", None)
        results.append(
            cash_magics._statement_processor.process_statement(
                ast.unparse(node), annotation=get_statement_annotations(code, node)
            )
        )
        assert len(cash_magics.shell.user_ns["frames"]) == 1000
    return results


def test_many_small_tables_cheaper_to_build_than_to_store_are_not_stored(cash_magics, priced_at):
    found = priced_at(60.0)
    first, again = _twice(cash_magics, BUILD)

    assert found and set(found) == {1000}
    assert "tables would take" in (first.get("skipped_reason") or ""), first
    assert again["status"] == CacheStatus.COMPUTED


def test_under_persist_they_are_stored(cash_magics, priced_at):
    """The control: ``@cash:persist`` keeps any value, however dear its
    store, so the same statement is stored."""
    priced_at(60.0)
    first, again = _twice(cash_magics, BUILD + "  # @cash:persist")

    assert not first.get("skipped_reason"), first
    assert again["status"] == CacheStatus.RESTORED


def test_when_computing_them_costs_more_than_storing_they_are_stored(cash_magics, priced_at):
    """The control: the same tables, from a statement that took longer than
    storing them does."""
    found = priced_at(0.001)
    first, again = _twice(cash_magics, "frames = time.sleep(0.05) or [t.copy() for _ in range(1000)]")

    assert found and set(found) == {1000}
    assert not first.get("skipped_reason"), first
    assert again["status"] == CacheStatus.RESTORED


@pytest.mark.parametrize(
    ("copy", "dump"),
    [(0.1, 0.45), (0.4, 0.15)],
    ids=["the-next-statement-waits-for-the-pickling", "a-hit-copies-them-again"],
)
def test_a_store_cheaper_than_the_compute_is_refused_when_a_hit_saves_less(
    cash_magics, on_disk, priced_at, computed_in, copy, dump
):
    """Copying and pickling take less than the 1 s compute (0.55 s), which
    let the store through before. With the next statement's wait for the
    pickling, or with a hit from RAM copying the tables again, the store
    costs more than a hit saves."""
    priced_at(copy=copy, dump=dump)
    computed_in(1.0)
    first, again = _twice(cash_magics, BUILD)

    assert "tables would take" in (first.get("skipped_reason") or ""), first
    assert again["status"] == CacheStatus.COMPUTED


def test_a_store_a_hit_pays_back_is_kept(cash_magics, on_disk, priced_at, computed_in):
    """The control: copying for 0.1 s and pickling for 0.3 s, twice that
    with the wait, is under the 1 s compute less the 0.1 s restore."""
    priced_at(copy=0.1, dump=0.3)
    computed_in(1.0)
    first, again = _twice(cash_magics, BUILD)

    assert not first.get("skipped_reason"), first
    assert again["status"] == CacheStatus.RESTORED


#: The first pass sleeps, so the loop clears the floors that leave a quick
#: loop unstored whatever its store costs.
LOOP = "for i in range(1000):\n    frames.append(time.sleep(0.15 if i == 0 else 0) or t.copy())"


def _walks(monkeypatch) -> list[str]:
    """The in-place and share checks of a statement's outputs, as they run."""
    walked: list[str] = []
    refusal, share = in_place.refusal, StatementProcessor._refuse_unrestorable_outputs

    def counted_refusal(*args, **kwargs):
        walked.append("in_place")
        return refusal(*args, **kwargs)

    def counted_share(self, *args, **kwargs):
        walked.append("share")
        return share(self, *args, **kwargs)

    monkeypatch.setattr(in_place, "refusal", counted_refusal)
    monkeypatch.setattr(StatementProcessor, "_refuse_unrestorable_outputs", counted_share)
    return walked


def _loop_row(cash_magics):
    rows = [s for s in cash_magics._last_cell_metrics["statements"] if s["code"].startswith("for i in")]
    assert len(rows) == 1, cash_magics._last_cell_metrics["statements"]
    return rows[0]


def test_a_refused_store_skips_the_walks_that_decide_whether_it_can_be_stored(
    cash_magics, priced_at, computed_in, monkeypatch
):
    """A loop appending the tables to a list is priced before its in-place
    and share checks, and refused: neither walks the list, and what the
    loop built is the same."""
    priced_at(60.0)
    computed_in(1.0)
    run_cash_cell(cash_magics, SETUP + "\nframes = []")
    walked = _walks(monkeypatch)
    run_cash_cell(cash_magics, LOOP)

    assert "tables would take" in (_loop_row(cash_magics).get("skipped_reason") or "")
    assert walked == []
    frames = cash_magics.shell.user_ns["frames"]
    assert len(frames) == 1000 and all(f["n"].tolist() == [1, 2, 3] for f in frames)


def test_a_store_let_through_still_walks_them(cash_magics, priced_at, computed_in, monkeypatch):
    """The control: the same loop, priced cheap, has both checks run."""
    priced_at(0.001)
    computed_in(1.0)
    run_cash_cell(cash_magics, SETUP + "\nframes = []")
    walked = _walks(monkeypatch)
    run_cash_cell(cash_magics, LOOP)

    assert not _loop_row(cash_magics).get("skipped_reason"), _loop_row(cash_magics)
    assert "in_place" in walked and "share" in walked, walked
