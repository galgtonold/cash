"""A plot run keeps one run of its statements per condition.

The same statements run where the helper memos, the random streams or the
environment stand otherwise (a call written again further down, where the
memo already holds what the first call computed) are kept as entries of
their own, listed in one record most recently used first, so a later run at
either place finds its own. A run under a condition already listed takes
its place at the front. The runs listed are bounded together by the size a
kept plot run may show: past it the least recently used are dropped, and a
smaller one after a big one still fits. A run no tier holds any more is
dropped from the record, as it can never be served.
"""

from __future__ import annotations

from cash.notebook.ipython import plot_runs
from cash.notebook.ipython.plot_runs import PlotRuns
from cash.notebook.plot_outputs import PLOT_RECORD_CAP

RECORD = "plot:record"


class _Backend:
    """What the record is read and written through: metadata looked at
    without counting a use, metadata-only writes, and deletes."""

    def __init__(self) -> None:
        self.meta: dict[str, dict] = {}
        self.deleted: list[str] = []

    def peek_metadata(self, key):
        found = self.meta.get(key)
        return dict(found) if found is not None else None

    def set_metadata_only(self, key, metadata):
        self.meta[key] = dict(metadata)

    def delete(self, key):
        self.deleted.append(key)
        self.meta.pop(key, None)

    def get(self, key):
        raise AssertionError("keeping a run read a kept run's payload")

    get_metadata = get


def _key(condition: str) -> str:
    return f"plot:{condition}"


def _keep(backend: _Backend, condition: str, size: int = 10) -> list[str]:
    backend.meta[_key(condition)] = {"plot_format": plot_runs._FORMAT}
    PlotRuns._note_variant(backend, RECORD, condition, size, _key)
    return [c for c, _ in plot_runs._listed_variants(backend, RECORD)]


def test_runs_under_other_conditions_are_listed_newest_first():
    backend = _Backend()
    for condition in "abcdef":
        listed = _keep(backend, condition)
    assert listed == list("fedcba"), "a run under a sixth condition dropped one that fits"


def test_a_run_under_a_listed_condition_moves_to_the_front():
    backend = _Backend()
    _keep(backend, "a")
    _keep(backend, "b")
    assert _keep(backend, "a") == ["a", "b"]


def test_past_the_size_the_least_recently_used_are_dropped_and_a_small_one_still_fits():
    backend = _Backend()
    _keep(backend, "small", size=10)
    _keep(backend, "big", size=PLOT_RECORD_CAP // 2 + 1)
    listed = _keep(backend, "new", size=PLOT_RECORD_CAP // 2 + 1)
    assert listed == ["new", "small"]
    assert backend.deleted == [_key("big")]


def test_a_run_no_tier_holds_is_dropped_from_the_record():
    backend = _Backend()
    _keep(backend, "a")
    _keep(backend, "b")
    del backend.meta[_key("a")]
    assert _keep(backend, "c") == ["c", "b"]


def test_a_served_run_is_listed_first():
    backend = _Backend()
    for condition in "abc":
        _keep(backend, condition)
    plots = PlotRuns(None, None, None)
    plots._promote = (backend, RECORD, "a")
    plots._used_first()
    assert [c for c, _ in plot_runs._listed_variants(backend, RECORD)] == ["a", "c", "b"]
    assert plots._promote is None
