"""Helper modules imported across cells, edited together with the cells that use them."""

import textwrap
import time

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexModulePatterns:
    """Test intricate module import and usage patterns."""

    @pytest.mark.modules
    def test_module_with_class_and_function(self, nb_runner, tmp_path):
        """Module provides both a class and a function."""
        mod = tmp_path / "helpers.py"
        mod.write_text(
            textwrap.dedent("""\
            class Processor:
                def __init__(self, factor):
                    self.factor = factor
                def process(self, x):
                    return x * self.factor
            
            def quick_process(x):
                return x * 2
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport helpers",
                "p = helpers.Processor(3)",
                "r1 = p.process(10)\nr2 = helpers.quick_process(10)",
                "print(f'Class: {r1}, Func: {r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Class: 30" in out
        assert "Func: 20" in out

    @pytest.mark.modules
    def test_module_with_global_state(self, nb_runner, tmp_path):
        """Module has module-level state (counter)."""
        mod = tmp_path / "stateful_mod.py"
        mod.write_text(
            textwrap.dedent("""\
            _count = 0
            def increment():
                global _count
                _count += 1
                return _count
            def get_count():
                return _count
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport stateful_mod",
                "r1 = stateful_mod.increment()",
                "r2 = stateful_mod.increment()",
                "count = stateful_mod.get_count()\nprint(f'Count: {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Count: 2" in out

    @pytest.mark.modules
    def test_two_modules_interacting(self, nb_runner, tmp_path):
        """Two local modules where one depends on the other."""
        base_mod = tmp_path / "base_utils.py"
        base_mod.write_text(
            textwrap.dedent("""\
            def normalize(x):
                return x / 100.0
        """),
            encoding="utf-8",
        )
        calc_mod = tmp_path / "calculator.py"
        calc_mod.write_text(
            textwrap.dedent("""\
            from base_utils import normalize
            def calc(x):
                return normalize(x) * 2
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport calculator",
                "result = calculator.calc(500)",
                "print(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Result: 10.0" in out


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestModuleReloadCombined:
    """Module reload combined with various other features."""

    @pytest.mark.modules
    @pytest.mark.upstream
    def test_module_function_change_propagates_downstream(self, nb_runner, tmp_path):
        """Changing a module function should invalidate all downstream users."""
        mod_path = str(tmp_path / "mymod.py").replace("\\", "/")

        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("def transform(x): return x * 2\n")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{str(tmp_path).replace(chr(92), '/')}')",
                "import mymod",
                "a = mymod.transform(5)",
                "b = a + 10",
                "print(f'b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "b=20" in output  # transform(5)=10, 10+10=20

        # Modify module
        time.sleep(0.1)
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("def transform(x): return x * 3\n")

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "b=25" in output2  # transform(5)=15, 15+10=25

    @pytest.mark.modules
    def test_from_import_function_and_constant_mixed(self, nb_runner, tmp_path):
        """Module with both function and constant from-imports."""
        mod_path = str(tmp_path / "config_mod.py").replace("\\", "/")

        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("VERSION = '1.0'\ndef greet(name): return f'Hello {name} v{VERSION}'\n")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{str(tmp_path).replace(chr(92), '/')}')",
                "from config_mod import VERSION, greet",
                "msg = greet('World')",
                "print(f'msg={msg} version={VERSION}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "Hello World v1.0" in output
        assert "version=1.0" in output

        # Update module
        time.sleep(0.1)
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("VERSION = '2.0'\ndef greet(name): return f'Hi {name} v{VERSION}'\n")

        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert "Hi World v2.0" in output2
        assert "version=2.0" in output2


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestImportPlusFunctionEdit:
    """Import + function definition + function call, with edits."""

    def test_import_used_in_function_then_edit(self, nb_runner):
        """Import used inside function body, edit function."""
        nb_runner.create_notebook(
            [
                "import math",
                "def area(r):\n    return math.pi * r ** 2",
                "a = area(1)\nprint(f'a = {a:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 3.14" in nb_runner.get_output(3)

        # Change to volume
        nb_runner.set_cell_source(2, "def area(r):\n    return (4/3) * math.pi * r ** 3")
        nb_runner.run_all()
        assert "a = 4.19" in nb_runner.get_output(3)

    def test_switch_import_and_function(self, nb_runner):
        """Switch from one import to another, function changes too."""
        nb_runner.create_notebook(
            [
                "import math",
                "def compute(x):\n    return math.sqrt(x)",
                "result = compute(16)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "import json")
        nb_runner.set_cell_source(2, "def compute(x):\n    return len(json.dumps({'val': x}))")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # json.dumps({'val': 16}) -> '{"val": 16}' which is 11 chars
        assert "result = 11" in output


@pytest.mark.modules
class TestFromImportCascadeChain:
    """Test from-import with multi-cell cascade chains."""

    def test_from_import_used_in_three_cells(self, nb_runner, tmp_path):
        """
        from X import Y in cell 1, used in cells 2 and 3.
        Module change should invalidate both downstream cells.
        """
        mod_path = tmp_path / "calc.py"
        mod_path.write_text("def double(x):\n    return x * 2\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "from calc import double",
                "a = double(5)\nprint(f'a = {a}')",
                "b = double(a)\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "a = 10" in nb_runner.get_output(2)
        assert "b = 20" in nb_runner.get_output(3)

        # Change double to triple
        mod_path.write_text("def double(x):\n    return x * 3\n", encoding="utf-8")
        time.sleep(0.5)

        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        assert "a = 15" in out2, f"Cell 2: expected a=15, got: {out2}"
        assert "b = 45" in out3, f"Cell 3: expected b=45, got: {out3}"

    def test_from_import_multiple_names(self, nb_runner, tmp_path):
        """
        from X import Y, Z — both names should update when module changes.
        """
        mod_path = tmp_path / "ops.py"
        mod_path.write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "from ops import add, sub",
                "r1 = add(10, 3)\nr2 = sub(10, 3)\nprint(f'add={r1}, sub={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "add=13" in out, f"Got: {out}"
        assert "sub=7" in out, f"Got: {out}"

        # Change add to multiply
        mod_path.write_text("def add(a, b):\n    return a * b\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8")
        time.sleep(0.5)

        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "add=30" in out2, f"Expected add=30, got: {out2}"
        assert "sub=7" in out2, f"sub should still be 7, got: {out2}"


@pytest.mark.modules
class TestMultiModuleDependencies:
    """Test scenarios with multiple interdependent modules."""

    def test_two_modules_one_cell(self, nb_runner, tmp_path):
        """Import two different modules and use both."""
        mod_a = tmp_path / "mod_a.py"
        mod_a.write_text("def fa(x):\n    return x + 1\n", encoding="utf-8")
        mod_b = tmp_path / "mod_b.py"
        mod_b.write_text("def fb(x):\n    return x * 2\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "import mod_a\nimport mod_b",
                "r = mod_a.fa(mod_b.fb(5))\nprint(f'r = {r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "r = 11" in nb_runner.get_output(2)

        # Change mod_b
        mod_b.write_text("def fb(x):\n    return x * 3\n", encoding="utf-8")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "r = 16" in out, f"Expected r=16, got: {out}"

    def test_module_importing_module(self, nb_runner, tmp_path):
        """
        Module A imports Module B. Change B, both should update.
        """
        mod_b = tmp_path / "helper.py"
        mod_b.write_text("FACTOR = 10\n", encoding="utf-8")
        mod_a = tmp_path / "processor.py"
        mod_a.write_text("from helper import FACTOR\ndef process(x):\n    return x * FACTOR\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "import processor",
                "result = processor.process(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 50" in nb_runner.get_output(2)

        # Change the helper module's FACTOR
        mod_b.write_text("FACTOR = 100\n", encoding="utf-8")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        # This tests transitive dependency detection
        assert "result = 500" in out, f"Expected result=500 after helper change, got: {out}"
