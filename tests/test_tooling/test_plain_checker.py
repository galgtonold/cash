"""The plain checker (tools/plain_checker) reports each place cash and
plain Python differ, and none where they agree."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import tools.plain_checker as plain_checker
from tools.plain_checker import Outcome, check, notebook
from tools.plain_checker.kernel import assigned_names

ROOT = Path(__file__).resolve().parents[2]


def test_a_difference_is_reported_with_where_it_was(monkeypatch, tmp_path):
    def fake(shape, work, config):
        x = "1" if config is None or config == "notrace" else "2"
        return Outcome(seen={"run1": {"cell 0 output": "hi\n", "x": x}}, statuses={"run1": {0: ["COMPUTED"]}})

    monkeypatch.setattr("tools.plain_checker.kernel.run_notebook", fake)
    report = check(notebook("fake", ["x = 1"]), tmp_path)
    assert not report.ok
    assert {(config, where) for config, _run, where, *_ in report.mismatches} == {("trace", "x"), ("floors0", "x")}
    assert "cash='2' plain='1'" in report.text()


def test_cells_take_each_edit_from_its_run_on():
    shape = notebook("edits", ["a = 1", "b = a"], edits={3: {0: "a = 2"}})
    assert shape.cells_for(2) == ["a = 1", "b = a"]
    assert shape.cells_for(3) == shape.cells_for(4) == ["a = 2", "b = a"]


def test_the_names_compared_by_default_are_the_assigned_ones():
    cells = ["import os\nx, (y, z) = 1, (2, 3)", "def f():\n    q = 1\nfor i in range(2):\n    w = i", "x += 1\n_h = 0"]
    assert assigned_names(cells) == ["x", "y", "z", "i"]


def test_the_shape_files_load():
    for path in sorted((ROOT / "tools" / "plain_checker" / "shapes").glob("*.py")):
        shapes = runpy.run_path(str(path))["SHAPES"]
        assert shapes, path
        assert all(isinstance(s, plain_checker.Shape) for s in shapes), path


@pytest.mark.integration
@pytest.mark.fresh_kernel
@pytest.mark.timeout(600)
def test_the_smoke_shapes_match_plain_python(tmp_path):
    for shape in runpy.run_path(str(ROOT / "tools" / "plain_checker" / "shapes" / "smoke.py"))["SHAPES"]:
        report = check(shape, tmp_path / shape.name, ["trace"])
        report.assert_ok()
        assert report.statuses["trace"]["run1"], report.text(True)


def test_a_difference_a_fresh_kernel_also_gives_is_kept_apart(monkeypatch, tmp_path):
    """Plain draws on through Run All (1, 3); cash gives the fresh kernel's
    first answer again (1, 1): REBUILT, not BAD."""

    def fake(shape, work, config):
        second = "3" if config is None else "1"
        return Outcome(seen={"run1": {"u": "1"}, "run2": {"u": second}})

    monkeypatch.setattr("tools.plain_checker.kernel.run_notebook", fake)
    report = check(notebook("fake", ["u = next(G)"]), tmp_path, ["trace"])
    assert report.ok and report.verdict == "REBUILT", report.text()
    assert [(run, where) for _c, run, where, *_ in report.rebuilt] == [("run2", "u")]


def test_a_plan_of_its_own_names_each_run():
    shape = notebook("plan", ["a = 1", "b = a"], plan=["all", 1, "restart", "all", 1])
    assert shape.runs() == [
        ("run1", False, None),
        ("step2", False, [1]),
        ("restart+run1", True, None),
        ("restart+step4", False, [1]),
    ]
    assert [name for name, *_ in notebook("default", ["a = 1"]).runs()] == list(plain_checker.RUNS)


def test_module_state_as_a_restart_gives_it_is_kept_apart_from_a_rebuild(monkeypatch, tmp_path):
    def fake(shape, work, config):
        second = "2" if config is None else "1"
        return Outcome(seen={"run1": {"n": "1", "v": "1"}, "run2": {"n": second, "v": second}})

    monkeypatch.setattr("tools.plain_checker.kernel.run_notebook", fake)
    report = check(notebook("fake", ["n = f()"], module_writes=["n"]), tmp_path, ["trace"])
    assert [where for *_, where, _c, _p in report.as_restart] == ["n"]
    assert [where for *_, where, _c, _p in report.rebuilt] == ["v"]
    assert report.verdict == "REBUILT"
    only_n = check(notebook("fake", ["n = f()"], module_writes=["n", "v"]), tmp_path, ["trace"])
    assert only_n.verdict == "AS-RESTART" and only_n.ok


def test_a_cell_rerun_alone_may_answer_as_the_run_all_before_it(monkeypatch, tmp_path):
    """``rest = list(it)`` re-run alone: plain gives ``[]``; cash re-runs the
    cells above and gives the Run All's ``[2]``. Only a shape that says so
    accepts it, and only the Run All's value."""

    def fake(shape, work, config):
        alone = "[]" if config is None else "[2]"
        last = "[]" if config is None else "[9]"
        return Outcome(
            seen={
                "run1": {"rest": "[2]"},
                "step2": {"rest": alone},
                "restart+run1": {"rest": "[2]"},
                "restart+step4": {"rest": last},
            },
            statuses={
                "run1": {0: ["COMPUTED"], 1: ["COMPUTED"]},
                "step2": {1: ["COMPUTED", "COMPUTED", "COMPUTED"]},
                "restart+run1": {0: ["COMPUTED"], 1: ["COMPUTED"]},
                "restart+step4": {1: ["COMPUTED", "COMPUTED", "COMPUTED"]},
            },
        )

    monkeypatch.setattr("tools.plain_checker.kernel.run_notebook", fake)
    plan = ["all", 1, "restart", "all", 1]
    cells = ["it = iter([1, 2])\nnext(it)", "rest = list(it)"]
    accepted = check(
        notebook("fake", cells, plan=plan, module_writes=["rest"], alone_as_run_all=True), tmp_path, ["trace"]
    )
    assert [run for _c, run, *_ in accepted.as_restart] == ["step2"]
    assert [run for _c, run, *_ in accepted.mismatches] == ["restart+step4"]
    plain_plan = check(notebook("fake", cells, plan=plan, module_writes=["rest"]), tmp_path, ["trace"])
    assert [run for _c, run, *_ in plain_plan.mismatches] == ["step2", "restart+step4"]


def test_a_cell_rerun_alone_answering_as_the_run_all_must_show_the_rebuild(monkeypatch, tmp_path):
    """A stale hit gives the Run All's value too: a re-run alone that ran
    no cell above it is a difference, not a rebuild."""

    def fake(shape, work, config):
        return Outcome(
            seen={"run1": {"rest": "[2]"}, "step2": {"rest": "[]" if config is None else "[2]"}},
            statuses={"run1": {0: ["COMPUTED"], 1: ["COMPUTED"]}, "step2": {1: ["RESTORED"]}},
        )

    monkeypatch.setattr("tools.plain_checker.kernel.run_notebook", fake)
    cells = ["it = iter([1, 2])\nnext(it)", "rest = list(it)"]
    report = check(
        notebook("fake", cells, plan=["all", 1], module_writes=["rest"], alone_as_run_all=True), tmp_path, ["trace"]
    )
    assert [run for _c, run, *_ in report.mismatches] == ["step2"]
