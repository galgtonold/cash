"""Batch 108 – Import statement interaction tests.

Tests that exercise import statements combined with cell edits,
module reloads, and kernel restarts.
"""

import pytest

pytestmark = [pytest.mark.modules, pytest.mark.stress, pytest.mark.timeout(30)]


class TestImportAndCellEdits:
    """Import statements with cell edits."""

    def test_import_then_edit_usage_cell(self, nb_runner):
        """Import a module, edit the cell that uses it."""
        nb_runner.create_notebook([
            "import math",
            "val = math.sqrt(16)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 4.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "val = math.sqrt(25)\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 5.0" in nb_runner.get_output(2)

    def test_add_import_then_use(self, nb_runner):
        """Edit a cell to add an import, then use it downstream."""
        nb_runner.create_notebook([
            "x = 42",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

        # Change first cell to an import + computation
        nb_runner.set_cell_source(1, "import math\nx = int(math.pi * 10)")
        nb_runner.set_cell_source(2, "print(f'x = {x}')")
        nb_runner.run_all()
        assert "x = 31" in nb_runner.get_output(2)

    def test_from_import_edit(self, nb_runner):
        """from X import Y, then edit to import different name."""
        nb_runner.create_notebook([
            "from math import sqrt",
            "val = sqrt(9)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 3.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "from math import ceil")
        nb_runner.set_cell_source(2, "val = ceil(3.2)\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 4" in nb_runner.get_output(2)

    def test_import_rerun_idempotent(self, nb_runner):
        """Re-running import cell should be idempotent."""
        nb_runner.create_notebook([
            "import json",
            "data = json.dumps({'a': 1})\nprint(f'data = {data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert 'data = {"a": 1}' in nb_runner.get_output(2)

        # Re-run everything — should get same result
        nb_runner.run_all()
        assert 'data = {"a": 1}' in nb_runner.get_output(2)

    def test_import_after_restart(self, nb_runner):
        """After restart, imports should be re-executed."""
        nb_runner.create_notebook([
            "import os",
            "cwd = os.getcwd()\nprint(f'has_cwd = {bool(cwd)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "has_cwd = True" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "has_cwd = True" in nb_runner.get_output(2)

    def test_import_alias_edit(self, nb_runner):
        """Import with alias, then edit alias."""
        nb_runner.create_notebook([
            "import math as m",
            "val = m.floor(3.7)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 3" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "import math as m")
        nb_runner.set_cell_source(2, "val = m.ceil(3.7)\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 4" in nb_runner.get_output(2)


class TestMultipleImports:
    """Multiple imports across cells."""

    def test_two_imports_edit_one(self, nb_runner):
        """Two import cells, edit one and re-run."""
        nb_runner.create_notebook([
            "import math",
            "import json",
            "val = math.sqrt(json.loads('4'))\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 2.0" in nb_runner.get_output(3)

        # Change from math to different usage
        nb_runner.set_cell_source(
            3, "val = math.sqrt(json.loads('9'))\nprint(f'val = {val}')"
        )
        nb_runner.run_all()
        assert "val = 3.0" in nb_runner.get_output(3)

    def test_import_and_function_def(self, nb_runner):
        """Import used inside a function definition."""
        nb_runner.create_notebook([
            "import math",
            "def circle_area(r):\n    return math.pi * r ** 2",
            "area = circle_area(1)\nprint(f'area = {area:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 3.1416" in nb_runner.get_output(3)

        # Redefine function
        nb_runner.set_cell_source(
            2, "def circle_area(r):\n    return math.pi * r ** 2 * 2"
        )
        nb_runner.run_all()
        assert "area = 6.2832" in nb_runner.get_output(3)


class TestCustomModuleReload:
    """Custom module file changes + import."""

    def test_custom_module_edit(self, nb_runner, tmp_path):
        """Edit a custom module file, re-import should pick up changes."""
        mod_path = tmp_path / "mymod.py"
        mod_path.write_text("VALUE = 10\n")
        mod_path_str = str(mod_path.parent).replace("\\", "/")

        nb_runner.create_notebook([
            f"import sys\nsys.path.insert(0, '{mod_path_str}')",
            "import mymod\nval = mymod.VALUE\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)

        # Edit the module
        mod_path.write_text("VALUE = 99\n")

        # Restart for clean import
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 99" in nb_runner.get_output(2)

    def test_custom_module_function_edit(self, nb_runner, tmp_path):
        """Custom module with function, edit function body."""
        mod_path = tmp_path / "helpers.py"
        mod_path.write_text("def compute(x):\n    return x * 2\n")
        mod_path_str = str(mod_path.parent).replace("\\", "/")

        nb_runner.create_notebook([
            f"import sys\nsys.path.insert(0, '{mod_path_str}')",
            "from helpers import compute\nresult = compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit module function
        mod_path.write_text("def compute(x):\n    return x * 3\n")

        # Restart for clean import
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)
