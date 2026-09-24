"""Helper modules on disk that the notebook imports, edited between runs."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.modules
class TestSingleModuleImportReload:
    """Test single user module import and modification patterns."""

    def test_import_user_module(self, nb_runner, tmp_path):
        """Import a user-defined module from a .py file."""
        mod_path = tmp_path / "helpers.py"
        mod_path.write_text("def greet(name):\n    return f'Hello {name}'\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "import helpers",
                textwrap.dedent("""\
                result = helpers.greet('World')
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(3)

    def test_from_import_user_module(self, nb_runner, tmp_path):
        """From-import specific items from user module."""
        mod_path = tmp_path / "mathutils.py"
        mod_path.write_text("def square(x):\n    return x ** 2\n\ndef cube(x):\n    return x ** 3\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "from mathutils import square, cube",
                textwrap.dedent("""\
                r1 = square(5)
                r2 = cube(3)
                print(r1, r2)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25 27" in nb_runner.get_output(3)

    def test_module_function_change_detected(self, nb_runner, tmp_path):
        """Changing a module function should invalidate cache."""
        mod_path = tmp_path / "compute.py"
        mod_path.write_text("def calc(x):\n    return x * 2\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "import compute",
                textwrap.dedent("""\
                result = compute.calc(10)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        # Modify module
        mod_path.write_text("def calc(x):\n    return x * 10\n", encoding="utf-8")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

    def test_module_constant_change_detected(self, nb_runner, tmp_path):
        """Changing a module constant should invalidate cache."""
        mod_path = tmp_path / "config_mod.py"
        mod_path.write_text("VERSION = '1.0'\nDEBUG = False\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "import config_mod",
                textwrap.dedent("""\
                print(f"v{config_mod.VERSION} debug={config_mod.DEBUG}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v1.0 debug=False" in nb_runner.get_output(3)

        # Modify module
        mod_path.write_text("VERSION = '2.0'\nDEBUG = True\n", encoding="utf-8")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "v2.0 debug=True" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.modules
class TestMultiModuleDependencies:
    """Test multiple user modules with inter-dependencies."""

    def test_two_independent_modules(self, nb_runner, tmp_path):
        """Two independent modules imported in same notebook."""
        (tmp_path / "mod_a.py").write_text("A_VAL = 10\n", encoding="utf-8")
        (tmp_path / "mod_b.py").write_text("B_VAL = 20\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "import mod_a",
                "import mod_b",
                textwrap.dedent("""\
                total = mod_a.A_VAL + mod_b.B_VAL
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(4)

        # Change only mod_a
        (tmp_path / "mod_a.py").write_text("A_VAL = 100\n", encoding="utf-8")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "120" in nb_runner.get_output(4)

    def test_module_importing_another_module(self, nb_runner, tmp_path):
        """Module that imports another module."""
        (tmp_path / "base_mod.py").write_text("BASE = 5\n", encoding="utf-8")
        (tmp_path / "derived_mod.py").write_text("from base_mod import BASE\nDERIVED = BASE * 3\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "import derived_mod",
                textwrap.dedent("""\
                print(derived_mod.DERIVED)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.modules
class TestModuleFileDependency:
    """Test modules that also have file dependencies."""

    def test_module_with_csv_reader(self, nb_runner, tmp_path):
        """Module contains a function that reads a CSV, file tracked."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x\n1\n2\n3\n", encoding="utf-8")
        csv_path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{csv_path_str}')",
                textwrap.dedent("""\
                total = df['x'].sum()
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(3)

        # Change CSV
        csv_path.write_text("x\n10\n20\n30\n", encoding="utf-8")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.modules
class TestModuleReloadEdgeCases:
    """Test tricky module reload scenarios."""

    def test_from_import_function_reload(self, nb_runner, tmp_path):
        """From-import a function, modify module, function should update."""
        mod_path = tmp_path / "toolbox.py"
        mod_path.write_text("def tool(x):\n    return x + 1\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "from toolbox import tool",
                textwrap.dedent("""\
                result = tool(10)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "11" in nb_runner.get_output(3)

        # Modify module function and restart kernel (sys.modules must be cleared)
        mod_path.write_text("def tool(x):\n    return x + 100\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "110" in nb_runner.get_output(3)

    def test_from_import_constant_reload(self, nb_runner, tmp_path):
        """From-import a constant, modify module, constant should update."""
        mod_path = tmp_path / "settings.py"
        mod_path.write_text("TIMEOUT = 30\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "from settings import TIMEOUT",
                textwrap.dedent("""\
                print(f"timeout={TIMEOUT}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "timeout=30" in nb_runner.get_output(3)

        # Modify constant and restart kernel (sys.modules must be cleared)
        mod_path.write_text("TIMEOUT = 60\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "timeout=60" in nb_runner.get_output(3)

    def test_module_add_new_function(self, nb_runner, tmp_path):
        """Add a new function to an existing module."""
        mod_path = tmp_path / "evolving.py"
        mod_path.write_text("def old_func():\n    return 'old'\n", encoding="utf-8")
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "import evolving",
                textwrap.dedent("""\
                result = evolving.old_func()
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "old" in nb_runner.get_output(3)

        # Add new function
        mod_path.write_text(
            "def old_func():\n    return 'old'\n\ndef new_func():\n    return 'new'\n", encoding="utf-8"
        )
        nb_runner.set_cell_source(
            3,
            textwrap.dedent("""\
            r1 = evolving.old_func()
            r2 = evolving.new_func()
            print(r1, r2)
        """),
        )
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "old" in output
        assert "new" in output

    def test_module_class_change(self, nb_runner, tmp_path):
        """Change a class defined in a module."""
        mod_path = tmp_path / "models.py"
        mod_path.write_text(
            "class Item:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "    def label(self):\n"
            "        return self.name.upper()\n",
            encoding="utf-8",
        )
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{path_str}')",
                "from models import Item",
                textwrap.dedent("""\
                item = Item('widget')
                print(item.label())
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "WIDGET" in nb_runner.get_output(3)

        # Change class method and restart kernel (sys.modules must be cleared)
        mod_path.write_text(
            "class Item:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "    def label(self):\n"
            "        return f'[{self.name}]'\n",
            encoding="utf-8",
        )
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[widget]" in nb_runner.get_output(3)


@pytest.mark.stress
class TestMultiModuleBasics:
    """Test multi-file module systems."""

    def test_module_chain_import(self, nb_runner, tmp_path):
        """Module A imports B, B imports C — chain dependency."""
        pkg = tmp_path / "chain_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "mod_c.py").write_text("BASE = 10\ndef get_base(): return BASE\n", encoding="utf-8")
        (pkg / "mod_b.py").write_text(
            "from chain_pkg.mod_c import get_base\ndef double(): return get_base() * 2\n", encoding="utf-8"
        )
        (pkg / "mod_a.py").write_text(
            "from chain_pkg.mod_b import double\ndef compute(): return double() + 1\n", encoding="utf-8"
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
                textwrap.dedent("""\
                from chain_pkg.mod_a import compute
                result = compute()
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=21" in nb_runner.get_output(2)  # 10*2+1

    def test_package_with_init(self, nb_runner, tmp_path):
        """Package with __init__.py exporting symbols."""
        pkg = tmp_path / "mathlib"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            "from mathlib.ops import add, multiply\n__version__ = '1.0'\n", encoding="utf-8"
        )
        (pkg / "ops.py").write_text("def add(a, b): return a + b\ndef multiply(a, b): return a * b\n", encoding="utf-8")
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
                textwrap.dedent("""\
                import mathlib
                r1 = mathlib.add(3, 4)
                r2 = mathlib.multiply(5, 6)
                print(f"r1={r1} r2={r2} ver={mathlib.__version__}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=7 r2=30 ver=1.0" in nb_runner.get_output(2)

    def test_subpackage_imports(self, nb_runner, tmp_path):
        """Nested subpackages with cross-imports."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "__init__.py").write_text("", encoding="utf-8")

        utils = root / "utils"
        utils.mkdir()
        (utils / "__init__.py").write_text("", encoding="utf-8")
        (utils / "helpers.py").write_text("def fmt(x): return f'[{x}]'\n", encoding="utf-8")

        core = root / "core"
        core.mkdir()
        (core / "__init__.py").write_text("", encoding="utf-8")
        (core / "engine.py").write_text(
            "from project.utils.helpers import fmt\ndef process(data): return [fmt(d) for d in data]\n",
            encoding="utf-8",
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
                textwrap.dedent("""\
                from project.core.engine import process
                result = process([1, 2, 3])
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['[1]', '[2]', '[3]']" in nb_runner.get_output(2)


@pytest.mark.stress
class TestModuleReloadChain:
    """Test module reload propagation in multi-file setup."""

    def test_deep_module_reload(self, nb_runner, tmp_path):
        """Reload propagates through deep module chain."""
        pkg = tmp_path / "deep_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "base.py").write_text("FACTOR = 2\n", encoding="utf-8")
        (pkg / "middle.py").write_text(
            "from deep_pkg.base import FACTOR\ndef scaled(x): return x * FACTOR\n", encoding="utf-8"
        )
        (pkg / "top.py").write_text(
            "from deep_pkg.middle import scaled\ndef compute(x): return scaled(x) + 1\n", encoding="utf-8"
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
                textwrap.dedent("""\
                from deep_pkg.top import compute
                result = compute(5)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(2)  # 5*2+1

    def test_conditional_import(self, nb_runner, tmp_path):
        """Conditional import based on config."""
        pkg = tmp_path / "cond_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "fast.py").write_text("def process(x): return x * 10\n", encoding="utf-8")
        (pkg / "slow.py").write_text("def process(x): return x + 1\n", encoding="utf-8")
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
                USE_FAST = True
            """),
                textwrap.dedent("""\
                if USE_FAST:
                    from cond_pkg.fast import process
                else:
                    from cond_pkg.slow import process
                result = process(5)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(2)

    def test_relative_like_imports(self, nb_runner, tmp_path):
        """Simulate relative imports with explicit paths."""
        pkg = tmp_path / "rel_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "constants.py").write_text("PI = 3.14159\nE = 2.71828\n", encoding="utf-8")
        (pkg / "math_ops.py").write_text(
            "from rel_pkg.constants import PI, E\n"
            "def circle_area(r): return PI * r * r\n"
            "def exp_approx(x): return E ** x\n",
            encoding="utf-8",
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
                textwrap.dedent("""\
                from rel_pkg.math_ops import circle_area, exp_approx
                area = circle_area(5)
                exp_val = exp_approx(2)
                print(f"area={area:.3f} exp={exp_val:.3f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "area=78.540" in out
        assert "exp=" in out


@pytest.mark.stress
@pytest.mark.modules
@pytest.mark.timeout(30)
class TestCustomModuleReload:
    """Custom module file changes + import."""

    def test_custom_module_edit(self, nb_runner, tmp_path):
        """Edit a custom module file, re-import should pick up changes."""
        mod_path = tmp_path / "mymod.py"
        mod_path.write_text("VALUE = 10\n", encoding="utf-8")
        mod_path_str = str(mod_path.parent).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{mod_path_str}')",
                "import mymod\nval = mymod.VALUE\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)

        # Edit the module
        mod_path.write_text("VALUE = 99\n", encoding="utf-8")

        # Restart for clean import
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 99" in nb_runner.get_output(2)

    def test_custom_module_function_edit(self, nb_runner, tmp_path):
        """Custom module with function, edit function body."""
        mod_path = tmp_path / "helpers.py"
        mod_path.write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")
        mod_path_str = str(mod_path.parent).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{mod_path_str}')",
                "from helpers import compute\nresult = compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit module function
        mod_path.write_text("def compute(x):\n    return x * 3\n", encoding="utf-8")

        # Restart for clean import
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.modules
class TestDynamicImportPatterns:
    """Test caching with dynamic import patterns."""

    def test_reimport_after_change(self, nb_runner, tmp_path):
        """Module reimported after source change."""
        mod_file = tmp_path / "mymod.py"
        mod_file.write_text("VALUE = 100\n", encoding="utf-8")
        sys_path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{sys_path_str}')",
                "import mymod",
                "print(mymod.VALUE)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

        # Change module and restart
        mod_file.write_text("VALUE = 999\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "999" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.modules
class TestMultiModuleImportInteraction:
    """Test interactions between multiple imported modules."""

    def test_two_modules_interact(self, nb_runner, tmp_path):
        """Two custom modules interact across cells."""
        (tmp_path / "mod_a.py").write_text("def double(x): return x * 2\n", encoding="utf-8")
        (tmp_path / "mod_b.py").write_text("def format_result(val): return f'Result: {val}'\n", encoding="utf-8")
        sys_path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{sys_path_str}')",
                "import mod_a\nimport mod_b",
                textwrap.dedent("""\
                val = mod_a.double(21)
                msg = mod_b.format_result(val)
                print(msg)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Result: 42" in nb_runner.get_output(3)
