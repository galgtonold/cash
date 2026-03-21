"""
Batch 19: Complex multi-file module dependency patterns.

Tests how cash handles user-defined modules written to disk, imported across
cells, modified during a session, and reloaded. These are the most challenging
scenarios for the caching system as they involve file tracking, module reload,
and cross-cell dependency propagation simultaneously.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.modules]


# ============================================================
# Test Group 1: Single Module Import and Reload
# ============================================================

class TestSingleModuleImportReload:
    """Test single user module import and modification patterns."""

    def test_import_user_module(self, nb_runner, tmp_path):
        """Import a user-defined module from a .py file."""
        mod_path = tmp_path / "helpers.py"
        mod_path.write_text("def greet(name):\n    return f'Hello {name}'\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "import helpers",
            textwrap.dedent("""\
                result = helpers.greet('World')
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(3)

    def test_from_import_user_module(self, nb_runner, tmp_path):
        """From-import specific items from user module."""
        mod_path = tmp_path / "mathutils.py"
        mod_path.write_text("def square(x):\n    return x ** 2\n\ndef cube(x):\n    return x ** 3\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "from mathutils import square, cube",
            textwrap.dedent("""\
                r1 = square(5)
                r2 = cube(3)
                print(r1, r2)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25 27" in nb_runner.get_output(3)

    def test_module_function_change_detected(self, nb_runner, tmp_path):
        """Changing a module function should invalidate cache."""
        mod_path = tmp_path / "compute.py"
        mod_path.write_text("def calc(x):\n    return x * 2\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "import compute",
            textwrap.dedent("""\
                result = compute.calc(10)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        # Modify module
        mod_path.write_text("def calc(x):\n    return x * 10\n")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

    def test_module_constant_change_detected(self, nb_runner, tmp_path):
        """Changing a module constant should invalidate cache."""
        mod_path = tmp_path / "config_mod.py"
        mod_path.write_text("VERSION = '1.0'\nDEBUG = False\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "import config_mod",
            textwrap.dedent("""\
                print(f"v{config_mod.VERSION} debug={config_mod.DEBUG}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v1.0 debug=False" in nb_runner.get_output(3)

        # Modify module
        mod_path.write_text("VERSION = '2.0'\nDEBUG = True\n")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "v2.0 debug=True" in nb_runner.get_output(3)


# ============================================================
# Test Group 2: Multi-Module Dependencies
# ============================================================

class TestMultiModuleDependencies:
    """Test multiple user modules with inter-dependencies."""

    def test_two_independent_modules(self, nb_runner, tmp_path):
        """Two independent modules imported in same notebook."""
        (tmp_path / "mod_a.py").write_text("A_VAL = 10\n")
        (tmp_path / "mod_b.py").write_text("B_VAL = 20\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "import mod_a",
            "import mod_b",
            textwrap.dedent("""\
                total = mod_a.A_VAL + mod_b.B_VAL
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(4)

        # Change only mod_a
        (tmp_path / "mod_a.py").write_text("A_VAL = 100\n")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "120" in nb_runner.get_output(4)

    def test_module_importing_another_module(self, nb_runner, tmp_path):
        """Module that imports another module."""
        (tmp_path / "base_mod.py").write_text("BASE = 5\n")
        (tmp_path / "derived_mod.py").write_text(
            "from base_mod import BASE\n"
            "DERIVED = BASE * 3\n"
        )
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "import derived_mod",
            textwrap.dedent("""\
                print(derived_mod.DERIVED)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(3)


# ============================================================
# Test Group 3: Module + File Dependency Interaction
# ============================================================

class TestModuleFileDependency:
    """Test modules that also have file dependencies."""

    def test_module_with_csv_reader(self, nb_runner, tmp_path):
        """Module contains a function that reads a CSV, file tracked."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x\n1\n2\n3\n")
        csv_path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{csv_path_str}')",
            textwrap.dedent("""\
                total = df['x'].sum()
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(3)

        # Change CSV
        csv_path.write_text("x\n10\n20\n30\n")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(3)


# ============================================================
# Test Group 4: Module Reload Edge Cases
# ============================================================

class TestModuleReloadEdgeCases:
    """Test tricky module reload scenarios."""

    def test_from_import_function_reload(self, nb_runner, tmp_path):
        """From-import a function, modify module, function should update."""
        mod_path = tmp_path / "toolbox.py"
        mod_path.write_text("def tool(x):\n    return x + 1\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "from toolbox import tool",
            textwrap.dedent("""\
                result = tool(10)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "11" in nb_runner.get_output(3)

        # Modify module function and restart kernel (sys.modules must be cleared)
        mod_path.write_text("def tool(x):\n    return x + 100\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "110" in nb_runner.get_output(3)

    def test_from_import_constant_reload(self, nb_runner, tmp_path):
        """From-import a constant, modify module, constant should update."""
        mod_path = tmp_path / "settings.py"
        mod_path.write_text("TIMEOUT = 30\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "from settings import TIMEOUT",
            textwrap.dedent("""\
                print(f"timeout={TIMEOUT}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "timeout=30" in nb_runner.get_output(3)

        # Modify constant and restart kernel (sys.modules must be cleared)
        mod_path.write_text("TIMEOUT = 60\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "timeout=60" in nb_runner.get_output(3)

    def test_module_add_new_function(self, nb_runner, tmp_path):
        """Add a new function to an existing module."""
        mod_path = tmp_path / "evolving.py"
        mod_path.write_text("def old_func():\n    return 'old'\n")
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "import evolving",
            textwrap.dedent("""\
                result = evolving.old_func()
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "old" in nb_runner.get_output(3)

        # Add new function
        mod_path.write_text(
            "def old_func():\n    return 'old'\n\n"
            "def new_func():\n    return 'new'\n"
        )
        nb_runner.set_cell_source(3, textwrap.dedent("""\
            r1 = evolving.old_func()
            r2 = evolving.new_func()
            print(r1, r2)
        """))
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
            "        return self.name.upper()\n"
        )
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{path_str}')",
            "from models import Item",
            textwrap.dedent("""\
                item = Item('widget')
                print(item.label())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "WIDGET" in nb_runner.get_output(3)

        # Change class method and restart kernel (sys.modules must be cleared)
        mod_path.write_text(
            "class Item:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "    def label(self):\n"
            "        return f'[{self.name}]'\n"
        )
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[widget]" in nb_runner.get_output(3)
