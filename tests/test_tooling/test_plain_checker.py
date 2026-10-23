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
