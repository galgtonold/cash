"""A statement that misses in a loop's pass names what moved since it last
ran in that same pass, not the loop's own variables.

``for s ...: for l ...: path = ...; for file in sorted(os.listdir(path)):``
(mlbook ch08): the inner loop missed in its second to fourth passes, and the
badge said ``input changed: l, path``. It compared with the last run of the
statement, the pass before, where the loop's variables are always other
ones; the input that moved was another.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import time\ndef label(tag, i):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S} / 200)\n    return tag + str(i)\n"
CELL = (
    "acc = []\n"
    "for side in ('x', 'y'):\n"
    "    for part in ('a', 'b'):\n"
    "        tag = side + part\n"
    "        for i in range(300):\n"
    "            acc = acc + [label(tag, i) * times]\n"
)


@pytest.fixture
def unit_rows(cash_magics, monkeypatch):
    """Run *cells* in order; the badge rows of the inner loop of the last."""
    rendered: list = []
    real_render = cash_magics.badges.render
    monkeypatch.setattr(
        cash_magics.badges, "render", lambda metrics, **kw: rendered.append(list(metrics)) or real_render(metrics, **kw)
    )

    def run(*cells: str) -> list[dict]:
        for cell in cells:
            run_cash_cell(cash_magics, cell)
        return [m for m in rendered[-1] if "for i in range(300)" in (m.get("code") or "")]

    return run


def test_each_pass_names_the_input_that_moved(unit_rows):
    first = unit_rows(SETUP, "times = 1", CELL)
    assert [str(m.get("status")) for m in first] == ["COMPUTED"] * 4

    again = unit_rows("times = 2", CELL)

    assert [str(m.get("status")) for m in again] == ["COMPUTED"] * 4
    assert [m.get("miss_reason") for m in again] == ["input changed: times"] * 4


def test_a_pass_that_never_ran_here_names_nothing_of_another_pass(unit_rows):
    unit_rows(SETUP, "times = 1", CELL)

    again = unit_rows(CELL.replace("('a', 'b')", "('c',)"))

    assert len(again) == 2
    reasons = [str(m.get("miss_reason") or "") for m in again]
    assert not any(name in reason for reason in reasons for name in ("part", "tag", "side")), reasons


def test_the_reason_reads_the_same_pass_first_what_it_only_reads():
    from cash.notebook.statement.input_change import pass_input_change_reason

    before = {"df": "d1", "path": "p", "pbar": "b1"}
    assert (
        pass_input_change_reason(
            before, {"df": "d1", "path": "p", "pbar": "b2"}, ["df", "path", "pbar"], ["df", "pbar"]
        )
        == "input changed: pbar"
    )
    assert (
        pass_input_change_reason(
            before, {"df": "d2", "path": "q", "pbar": "b2"}, ["df", "path", "pbar"], ["df", "pbar"]
        )
        == "input changed: path"
    )
    assert pass_input_change_reason(before, dict(before), ["df", "path", "pbar"], ["df", "pbar"]) is None
    assert pass_input_change_reason(None, dict(before), ["df"], []) is None
