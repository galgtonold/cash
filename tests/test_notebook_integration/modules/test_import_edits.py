"""Editing import statements and aliases."""

import pytest

pytestmark = [pytest.mark.stress]


# Import statement interaction tests.
#
# Tests that exercise import statements combined with cell edits,
# module reloads, and kernel restarts.
@pytest.mark.modules
@pytest.mark.timeout(30)
class TestImportAndCellEdits:
    """Import statements with cell edits."""

    def test_import_then_edit_usage_cell(self, nb_runner):
        """Import a module, edit the cell that uses it."""
        nb_runner.create_notebook(
            [
                "import math",
                "val = math.sqrt(16)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 4.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "val = math.sqrt(25)\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 5.0" in nb_runner.get_output(2)

    def test_add_import_then_use(self, nb_runner):
        """Edit a cell to add an import, then use it downstream."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "print(f'x = {x}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from math import sqrt",
                "val = sqrt(9)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 3.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "from math import ceil")
        nb_runner.set_cell_source(2, "val = ceil(3.2)\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 4" in nb_runner.get_output(2)

    def test_import_rerun_idempotent(self, nb_runner):
        """Re-running import cell should be idempotent."""
        nb_runner.create_notebook(
            [
                "import json",
                "data = json.dumps({'a': 1})\nprint(f'data = {data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert 'data = {"a": 1}' in nb_runner.get_output(2)

        # Re-run everything — should get same result
        nb_runner.run_all()
        assert 'data = {"a": 1}' in nb_runner.get_output(2)

    def test_import_after_restart(self, nb_runner):
        """After restart, imports should be re-executed."""
        nb_runner.create_notebook(
            [
                "import os",
                "cwd = os.getcwd()\nprint(f'has_cwd = {bool(cwd)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "has_cwd = True" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "has_cwd = True" in nb_runner.get_output(2)

    def test_import_alias_edit(self, nb_runner):
        """Import with alias, then edit alias."""
        nb_runner.create_notebook(
            [
                "import math as m",
                "val = m.floor(3.7)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 3" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "import math as m")
        nb_runner.set_cell_source(2, "val = m.ceil(3.7)\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 4" in nb_runner.get_output(2)


# Import and module interaction tests.
#
# Tests where users import modules, edit imports, change which
# modules are used, and verify caching handles module changes.
@pytest.mark.modules
@pytest.mark.timeout(45)
class TestImportStatementEdits:
    """Edit import statements and verify downstream effects."""

    def test_change_imported_function(self, nb_runner):
        """Change from one math function to another."""
        nb_runner.create_notebook(
            [
                "from math import sqrt",
                "result = sqrt(144)\nprint(f'result = {result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 100",
                "result = x + 1\nprint(f'result = {result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import json as j",
                "data = j.dumps({'a': 1})\nprint(f'data = {data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "data = " in output

        # Change data being serialized
        nb_runner.set_cell_source(2, "data = j.dumps({'a': 1, 'b': 2})\nprint(f'data = {data}')")
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert '"b": 2' in output or '"b":2' in output


# Import alias and module-level function interaction tests.
#
# Tests editing import aliases, switching between import styles,
# and using module-level functions with edits.
@pytest.mark.modules
@pytest.mark.timeout(90)
class TestImportAliasEdits:
    """Editing import alias patterns."""

    def test_edit_import_alias(self, nb_runner):
        """Change an import alias and verify downstream uses."""
        nb_runner.create_notebook(
            [
                "import math as m  # alias v1",
                "result = m.sqrt(144)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12.0" in nb_runner.get_output(2)

        # Change to use different function
        nb_runner.set_cell_source(2, "result = m.factorial(5)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

    def test_switch_import_style(self, nb_runner):
        """Switch between import styles."""
        nb_runner.create_notebook(
            [
                "from os.path import join  # from import style",
                "result = join('a', 'b', 'c')\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a" in out and "b" in out and "c" in out

        # Switch to module import
        nb_runner.set_cell_source(1, "import os.path  # module import style")
        nb_runner.set_cell_source(2, "result = os.path.join('x', 'y', 'z')\nprint(f'result = {result}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "x" in out2 and "y" in out2 and "z" in out2


@pytest.mark.modules
@pytest.mark.timeout(30)
class TestMultipleImports:
    """Multiple imports across cells."""

    def test_two_imports_edit_one(self, nb_runner):
        """Two import cells, edit one and re-run."""
        nb_runner.create_notebook(
            [
                "import math",
                "import json",
                "val = math.sqrt(json.loads('4'))\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 2.0" in nb_runner.get_output(3)

        # Change from math to different usage
        nb_runner.set_cell_source(3, "val = math.sqrt(json.loads('9'))\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 3.0" in nb_runner.get_output(3)

    def test_import_and_function_def(self, nb_runner):
        """Import used inside a function definition."""
        nb_runner.create_notebook(
            [
                "import math",
                "def circle_area(r):\n    return math.pi * r ** 2",
                "area = circle_area(1)\nprint(f'area = {area:.4f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 3.1416" in nb_runner.get_output(3)

        # Redefine function
        nb_runner.set_cell_source(2, "def circle_area(r):\n    return math.pi * r ** 2 * 2")
        nb_runner.run_all()
        assert "area = 6.2832" in nb_runner.get_output(3)


@pytest.mark.modules
@pytest.mark.timeout(45)
class TestModuleReusePatterns:
    """Tests for using modules across multiple cells."""

    def test_use_module_in_two_cells_edit_one(self, nb_runner):
        """Import module, use in 2 cells, edit one."""
        nb_runner.create_notebook(
            [
                "import math",
                "a = math.floor(3.7)\nprint(f'a = {a}')",
                "b = math.ceil(3.2)\nprint(f'b = {b}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import math\ndef my_sqrt(x):\n    return math.sqrt(x)",
                "result = my_sqrt(25)\nprint(f'result = {result}')",
            ]
        )
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


@pytest.mark.modules
@pytest.mark.timeout(90)
class TestModuleFunctionEdits:
    """Editing usage of module-level functions."""

    def test_edit_module_function_args(self, nb_runner):
        """Edit arguments to module functions."""
        nb_runner.create_notebook(
            [
                "import json",
                "data = json.dumps({'a': 1})\nprint(f'data = {data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert 'data = {"a": 1}' in nb_runner.get_output(2)

        # Change to pretty print
        nb_runner.set_cell_source(
            2,
            "data = json.dumps({'a': 1, 'b': 2}, indent=2)\nprint(f'len = {len(data)}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len = " in out

    def test_edit_collections_usage(self, nb_runner):
        """Edit usage of collections module functions."""
        nb_runner.create_notebook(
            [
                "from collections import Counter",
                "c = Counter('aabbcc')\nprint(f'c = {dict(c)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 2" in nb_runner.get_output(2)

        # Change input
        nb_runner.set_cell_source(2, "c = Counter('aaabbb')\nprint(f'c = {dict(c)}')")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 3" in out
        assert "'b': 3" in out

    def test_edit_itertools_usage(self, nb_runner):
        """Edit itertools pipeline."""
        nb_runner.create_notebook(
            [
                "from itertools import chain, repeat",
                "result = list(chain([1, 2], [3, 4]))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to repeat
        nb_runner.set_cell_source(2, "result = list(repeat(42, 3))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [42, 42, 42]" in nb_runner.get_output(2)


@pytest.mark.modules
@pytest.mark.timeout(45)
class TestConditionalImportSwitching:
    """Tests with conditional import patterns."""

    def test_switch_between_json_modes(self, nb_runner):
        """Switch between json and string formatting."""
        nb_runner.create_notebook(
            [
                "import json\nuse_json = True",
                "data = {'key': 'value', 'num': 42}",
                "if use_json:\n    output = json.dumps(data)\nelse:\n    output = str(data)\nprint(f'output = {output}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "output = " in output

        # Switch to non-json
        nb_runner.set_cell_source(1, "import json\nuse_json = False")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "output = " in output


# Conditional import and lazy loading interaction tests.
# Tests that editing code with conditional imports and lazy attribute
# access properly invalidates downstream cells.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestConditionalImportInteraction:
    """Test conditional import/lazy loading with cache invalidation."""

    def test_conditional_import_flag_edit(self, nb_runner):
        """Editing a flag that controls conditional import should propagate."""
        nb_runner.create_notebook(
            [
                "use_math = True",
                "if use_math:\n    from math import pi\n    val = round(pi, 4)\nelse:\n    val = 3.0",
                "result = val * 2",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=6.2832" in out

        nb_runner.set_cell_source(1, "use_math = False")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=6.0" in out

    def test_try_import_fallback_edit(self, nb_runner):
        """Editing try/except import fallback should propagate."""
        nb_runner.create_notebook(
            [
                "module_name = 'math'",
                (
                    "if module_name == 'math':\n"
                    "    import math\n"
                    "    sqrt_fn = math.sqrt\n"
                    "else:\n"
                    "    sqrt_fn = lambda x: x ** 0.5"
                ),
                "result = round(sqrt_fn(144), 2)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=12.0" in out

        nb_runner.set_cell_source(1, "module_name = 'none'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=12.0" in out  # Both paths give same result for 144

    def test_lazy_attribute_access_edit(self, nb_runner):
        """Editing a lazy-loaded object should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Lazy:\n"
                    "    def __init__(self, factory):\n"
                    "        self._factory = factory\n"
                    "        self._value = None\n"
                    "    @property\n"
                    "    def value(self):\n"
                    "        if self._value is None:\n"
                    "            self._value = self._factory()\n"
                    "        return self._value"
                ),
                "lazy = Lazy(lambda: list(range(5)))",
                "result = sum(lazy.value)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "lazy = Lazy(lambda: list(range(10)))")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=45" in out
