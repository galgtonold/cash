"""Batch 139 – Import and module interaction tests.

Tests where users import modules, edit imports, change which
modules are used, and verify caching handles module changes.
"""

import pytest

pytestmark = [pytest.mark.modules, pytest.mark.stress, pytest.mark.timeout(45)]


class TestImportStatementEdits:
    """Edit import statements and verify downstream effects."""

    def test_change_imported_function(self, nb_runner):
        """Change from one math function to another."""
        nb_runner.create_notebook([
            "from math import sqrt",
            "result = sqrt(144)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12.0" in nb_runner.get_output(2)

        # Switch to different function
        nb_runner.set_cell_source(1, "from math import log2")
        nb_runner.set_cell_source(2, "result = log2(256)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 8.0" in nb_runner.get_output(2)

    def test_add_import_use_it(self, nb_runner):
        """Add a new import and use it in existing cell."""
        nb_runner.create_notebook([
            "x = 100",
            "result = x + 1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 101" in nb_runner.get_output(2)

        # Add import and use it
        nb_runner.set_cell_source(1, "import math\nx = 100")
        nb_runner.set_cell_source(2, "result = int(math.sqrt(x))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

    def test_import_alias_change(self, nb_runner):
        """Change import alias."""
        nb_runner.create_notebook([
            "import json as j",
            "data = j.dumps({'a': 1})\nprint(f'data = {data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "data = " in output

        # Change data being serialized
        nb_runner.set_cell_source(
            2, "data = j.dumps({'a': 1, 'b': 2})\nprint(f'data = {data}')"
        )
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert '"b": 2' in output or '"b":2' in output


class TestModuleReusePatterns:
    """Tests for using modules across multiple cells."""

    def test_use_module_in_two_cells_edit_one(self, nb_runner):
        """Import module, use in 2 cells, edit one."""
        nb_runner.create_notebook([
            "import math",
            "a = math.floor(3.7)\nprint(f'a = {a}')",
            "b = math.ceil(3.2)\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 3" in nb_runner.get_output(2)
        assert "b = 4" in nb_runner.get_output(3)

        # Edit cell 2 only
        nb_runner.set_cell_source(2, "a = math.floor(9.9)\nprint(f'a = {a}')")
        nb_runner.run_all()
        assert "a = 9" in nb_runner.get_output(2)
        assert "b = 4" in nb_runner.get_output(3)

    def test_stdlib_to_custom_function(self, nb_runner):
        """Replace stdlib call with custom function."""
        nb_runner.create_notebook([
            "import math\ndef my_sqrt(x):\n    return math.sqrt(x)",
            "result = my_sqrt(25)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5.0" in nb_runner.get_output(2)

        # Replace with custom
        nb_runner.set_cell_source(
            1,
            "def my_sqrt(x):\n    return x ** 0.5  # no math import",
        )
        nb_runner.run_all()
        assert "result = 5.0" in nb_runner.get_output(2)


class TestConditionalImportSwitching:
    """Tests with conditional import patterns."""

    def test_switch_between_json_modes(self, nb_runner):
        """Switch between json and string formatting."""
        nb_runner.create_notebook([
            "import json\nuse_json = True",
            "data = {'key': 'value', 'num': 42}",
            "if use_json:\n    output = json.dumps(data)\nelse:\n    output = str(data)\nprint(f'output = {output}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "output = " in output

        # Switch to non-json
        nb_runner.set_cell_source(1, "import json\nuse_json = False")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "output = " in output
