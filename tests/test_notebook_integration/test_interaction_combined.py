"""Batch 112 – Complex combined interaction tests.

Tests that combine multiple features: functions + files, imports + edits + restart,
loops + mutations + edits, etc. These simulate real-world notebook workflows.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(30)]


class TestFunctionPlusFileEdit:
    """Function that reads file, both function and file edited."""

    def test_function_reads_file_then_file_changes(self, nb_runner, tmp_path):
        """Function reads a file; file content changes."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("100")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"def load_value():\n    with open('{path_str}') as f:\n        return int(f.read().strip())",
            "val = load_value()\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 100" in nb_runner.get_output(2)

        # Change file
        data_file.write_text("999")
        nb_runner.run_all()
        assert "val = 999" in nb_runner.get_output(2)

    def test_function_and_file_both_change(self, nb_runner, tmp_path):
        """Both the function definition and the file change."""
        data_file = tmp_path / "vals.txt"
        data_file.write_text("10\n20\n30")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"def process():\n    with open('{path_str}') as f:\n        return sum(int(x) for x in f)",
            "result = process()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        # Change file content and function
        data_file.write_text("1\n2\n3")
        nb_runner.set_cell_source(
            1,
            f"def process():\n    with open('{path_str}') as f:\n        return max(int(x) for x in f)",
        )
        nb_runner.run_all()
        assert "result = 3" in nb_runner.get_output(2)


class TestFunctionPlusLoopEdit:
    """Function used inside a loop, both edited."""

    def test_edit_function_used_in_loop(self, nb_runner):
        """Function called inside loop, redefine function."""
        nb_runner.create_notebook([
            "def transform(x):\n    return x * 2",
            "results = []\nfor i in range(4):\n    results.append(transform(i))",
            "print(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [0, 2, 4, 6]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "results = [0, 1, 4, 9]" in nb_runner.get_output(3)

    def test_edit_loop_and_function(self, nb_runner):
        """Edit both the loop and the function it uses."""
        nb_runner.create_notebook([
            "def scale(x, factor):\n    return x * factor",
            "out = []\nfor v in [1, 2, 3]:\n    out.append(scale(v, 10))",
            "print(f'out = {out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = [10, 20, 30]" in nb_runner.get_output(3)

        # Edit function and loop data
        nb_runner.set_cell_source(
            1, "def scale(x, factor):\n    return x + factor"
        )
        nb_runner.set_cell_source(
            2, "out = []\nfor v in [10, 20, 30]:\n    out.append(scale(v, 100))"
        )
        nb_runner.run_all()
        assert "out = [110, 120, 130]" in nb_runner.get_output(3)


class TestImportPlusFunctionEdit:
    """Import + function definition + function call, with edits."""

    def test_import_used_in_function_then_edit(self, nb_runner):
        """Import used inside function body, edit function."""
        nb_runner.create_notebook([
            "import math",
            "def area(r):\n    return math.pi * r ** 2",
            "a = area(1)\nprint(f'a = {a:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 3.14" in nb_runner.get_output(3)

        # Change to volume
        nb_runner.set_cell_source(
            2, "def area(r):\n    return (4/3) * math.pi * r ** 3"
        )
        nb_runner.run_all()
        assert "a = 4.19" in nb_runner.get_output(3)

    def test_switch_import_and_function(self, nb_runner):
        """Switch from one import to another, function changes too."""
        nb_runner.create_notebook([
            "import math",
            "def compute(x):\n    return math.sqrt(x)",
            "result = compute(16)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "import json")
        nb_runner.set_cell_source(
            2, "def compute(x):\n    return len(json.dumps({'val': x}))"
        )
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # json.dumps({'val': 16}) -> '{"val": 16}' which is 11 chars
        assert "result = 11" in output


class TestDataPipelineEdits:
    """Simulate a data processing pipeline with edits at various stages."""

    def test_etl_pipeline_edit_transform(self, nb_runner):
        """ETL: extract -> transform -> load. Edit transform step."""
        nb_runner.create_notebook([
            "raw = [1, 2, 3, 4, 5]",
            "transformed = [x * 2 for x in raw]",
            "result = sum(transformed)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit transform
        nb_runner.set_cell_source(2, "transformed = [x ** 2 for x in raw]")
        nb_runner.run_all()
        assert "result = 55" in nb_runner.get_output(3)

    def test_etl_pipeline_edit_source(self, nb_runner):
        """ETL: edit the source data."""
        nb_runner.create_notebook([
            "raw = [10, 20, 30]",
            "filtered = [x for x in raw if x > 15]",
            "total = sum(filtered)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 50" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "raw = [5, 10, 15, 20, 25, 30]")
        nb_runner.run_all()
        assert "total = 75" in nb_runner.get_output(3)

    def test_multi_stage_pipeline_edit_multiple(self, nb_runner):
        """Multi-stage pipeline: source -> filter -> transform -> aggregate."""
        nb_runner.create_notebook([
            "data = list(range(1, 11))",
            "filtered = [x for x in data if x % 2 == 0]",
            "transformed = [x * 10 for x in filtered]",
            "result = sum(transformed)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(4)

        # Edit filter and transform
        nb_runner.set_cell_source(
            2, "filtered = [x for x in data if x % 2 != 0]"
        )
        nb_runner.set_cell_source(3, "transformed = [x * 100 for x in filtered]")
        nb_runner.run_all()
        assert "result = 2500" in nb_runner.get_output(4)


class TestConditionalChainEdits:
    """Conditional logic chains with edits."""

    def test_nested_conditionals_edit(self, nb_runner):
        """Nested if-else, edit the input to change branch taken."""
        nb_runner.create_notebook([
            "score = 85",
            "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelse:\n    grade = 'C'",
            "print(f'grade = {grade}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade = B" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "grade = A" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "score = 50")
        nb_runner.run_all()
        assert "grade = C" in nb_runner.get_output(3)

    def test_conditional_chain_edit_thresholds(self, nb_runner):
        """Edit the thresholds in conditionals."""
        nb_runner.create_notebook([
            "val = 50",
            "if val > 100:\n    cat = 'high'\nelse:\n    cat = 'low'",
            "print(f'cat = {cat}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cat = low" in nb_runner.get_output(3)

        # Lower threshold
        nb_runner.set_cell_source(
            2, "if val > 40:\n    cat = 'high'\nelse:\n    cat = 'low'"
        )
        nb_runner.run_all()
        assert "cat = high" in nb_runner.get_output(3)
