"""The core set holds a share of every feature folder's tests.

The set is picked by cover: a test is in when it runs a line, a feature
pair or a step order no picked test runs. That says each line ran, not that
each behaviour was asserted. The chart folder had no test in the set at all
and the language folder 42 of 1,400, so a change that kept the same lines
running and gave another value passed a push. A floor per folder
(``select_core.FLOOR``) is added on top of the cover.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.test_selection.select_core import FLOOR, below_floor, collected_tests, floor_of, folder_of, top_up

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
CORE_SET = REPO_ROOT / "tools" / "test_selection" / "core_set.txt"
BASE = "tests/test_notebook_integration"


def _tests(folder: str, file: str, count: int) -> list[str]:
    return [f"{BASE}/{folder}/{file}.py::test_{i:02d}" for i in range(count)]


@pytest.mark.timeout(180)
def test_the_committed_core_set_holds_the_floor_of_every_folder():
    chosen = [line.strip() for line in CORE_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    collected = collected_tests()
    assert len(collected) > 1000, "the integration suite was not collected: the check would pass on nothing"
    short = below_floor(chosen, collected)
    assert not short, (
        f"folders with fewer than {FLOOR:.0%} of their tests in the core set (held, floor): {short}\n"
        "Run `python tools/test_selection/select_core.py --top-up`, then the tests it adds."
    )


def test_the_list_is_kept_sorted_so_an_addition_is_one_line_of_diff():
    lines = CORE_SET.read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines)


def test_a_folder_with_no_pick_is_topped_up_to_its_floor():
    charts = _tests("charts", "test_bars", 30) + _tests("charts", "test_lines", 30)
    loops = _tests("loops", "test_for", 40)
    chosen = loops[:5]
    added = top_up(chosen, charts + loops)
    assert [folder_of(n) for n in added] == ["charts"] * floor_of(60) == ["charts"] * 3
    assert below_floor([*chosen, *added], charts + loops) == {}
    assert top_up([*chosen, *added], charts + loops) == []


def test_additions_spread_over_a_folders_files_before_one_file_gives_two():
    tests = _tests("language", "test_a", 20) + _tests("language", "test_b", 20) + _tests("language", "test_c", 20)
    added = top_up([], tests, share=0.1)
    files = [n.split("::")[0].rsplit("/", 1)[1] for n in added]
    assert sorted(files[:3]) == sorted(files[3:]) == ["test_a.py", "test_b.py", "test_c.py"]


def test_the_cheapest_tests_of_a_turn_are_taken_first():
    tests = _tests("charts", "test_bars", 20) + _tests("charts", "test_lines", 20)
    cost = dict.fromkeys(tests, 9.0)
    cost[tests[7]] = 1.0  # test_bars, test_07
    cost[tests[25]] = 2.0  # test_lines, test_05
    assert top_up([], tests, cost, share=0.05) == [tests[7], tests[25]]
    assert top_up([], tests, cost, share=0.025) == [tests[7]]


def test_a_test_already_chosen_counts_toward_the_floor_and_is_not_added_again():
    tests = _tests("charts", "test_bars", 40)
    assert top_up(tests[:1], tests) == [tests[1]]
    assert top_up(tests[:2], tests) == []


def test_a_folder_never_needs_more_than_it_has():
    assert floor_of(0) == 0
    assert floor_of(1) == 1
    assert floor_of(20) == 1
    assert floor_of(21) == 2
    tests = _tests("harness", "test_only", 3)
    assert top_up([], tests, share=2.0) == tests
