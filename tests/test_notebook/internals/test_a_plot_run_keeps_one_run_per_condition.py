"""A kept plot run holds one run of its statements per condition.

The same statements run where the helper memos, the random streams or the
environment stand otherwise (a call written again further down, where the
memo already holds what the first call computed) keep their runs side by
side under the one key, so a later run at either place finds its own. A
run under a condition already kept replaces that one. How many are kept is
bounded by count and by the size a kept plot run may show.
"""

from __future__ import annotations

from cash.notebook.ipython import plot_runs
from cash.notebook.ipython.plot_runs import PlotRuns
from cash.notebook.plot_outputs import PLOT_RECORD_CAP

KEY = "plot:abc"


class _Backend:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[dict, dict]] = {}

    def get(self, key):
        return self.entries.get(key, (None, None))

    def set(self, key, payload, metadata):
        self.entries[key] = (metadata, payload)


def _variant(memo: str, size: int = 10, tag: str = "") -> dict:
    return {
        "statements": [{"code": "run()", "memos": [[[{"where": ["random_cities"], "items": [memo]}], None]]}],
        "flush": None,
        "echo": tag,
        "bound": [],
        "env": {},
        "rng_reads": {},
        "seconds": 1.0,
        "size": size,
    }


def _keep(backend: _Backend, variant: dict) -> list[dict]:
    variants = PlotRuns._variants_with(backend, KEY, variant)
    backend.set(KEY, {"variants": variants}, {"plot_format": plot_runs._FORMAT})
    return variants


def test_a_run_under_another_memo_is_kept_beside_the_first():
    backend = _Backend()
    _keep(backend, _variant("filled", tag="first place"))
    kept = _keep(backend, _variant("read", tag="second place"))
    assert [v["echo"] for v in kept] == ["second place", "first place"]


def test_a_run_under_a_kept_condition_replaces_it():
    backend = _Backend()
    _keep(backend, _variant("filled", tag="old"))
    _keep(backend, _variant("read"))
    kept = _keep(backend, _variant("filled", tag="new"))
    assert [v["echo"] for v in kept] == ["new", ""]


def test_the_oldest_conditions_are_dropped_past_the_count():
    backend = _Backend()
    for n in range(plot_runs._VARIANTS_KEPT + 2):
        kept = _keep(backend, _variant(str(n), tag=str(n)))
    newest = [str(n) for n in reversed(range(plot_runs._VARIANTS_KEPT + 2))]
    assert [v["echo"] for v in kept] == newest[: plot_runs._VARIANTS_KEPT]


def test_runs_past_the_size_a_plot_may_show_are_dropped():
    backend = _Backend()
    _keep(backend, _variant("a", size=PLOT_RECORD_CAP // 2 + 1, tag="a"))
    kept = _keep(backend, _variant("b", size=PLOT_RECORD_CAP // 2 + 1, tag="b"))
    assert [v["echo"] for v in kept] == ["b"]


def test_an_entry_of_another_format_is_not_read():
    backend = _Backend()
    backend.set(KEY, {"statements": []}, {"plot_format": plot_runs._FORMAT - 1})
    assert [v["echo"] for v in PlotRuns._variants_with(backend, KEY, _variant("a", tag="a"))] == ["a"]
