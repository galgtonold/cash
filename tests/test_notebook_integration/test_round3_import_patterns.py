"""
Batch 32: Import patterns — dynamic imports, conditional imports, importlib,
sys.path manipulation, and star imports across cells.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.modules]


class TestDynamicImportPatterns:
    """Test caching with dynamic import patterns."""

    def test_importlib_import_module(self, nb_runner):
        """importlib.import_module across cells."""
        nb_runner.create_notebook([
            "import importlib",
            textwrap.dedent("""\
                math_mod = importlib.import_module('math')
                result = math_mod.sqrt(144)
                print(f"sqrt={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sqrt=12.0" in nb_runner.get_output(2)

    def test_conditional_import(self, nb_runner):
        """Conditional import based on availability."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                try:
                    import numpy as np
                    HAS_NUMPY = True
                except ImportError:
                    HAS_NUMPY = False
            """),
            textwrap.dedent("""\
                if HAS_NUMPY:
                    arr = np.array([1, 2, 3])
                    print(f"sum={arr.sum()}")
                else:
                    print("no numpy")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=6" in nb_runner.get_output(2)

    def test_import_alias_chain(self, nb_runner):
        """Import with alias used across multiple cells."""
        nb_runner.create_notebook([
            "import collections as col",
            "c = col.Counter('abracadabra')",
            "print(c.most_common(3))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'a'" in output

    def test_from_import_multiple_items(self, nb_runner):
        """from X import a, b, c across cells."""
        nb_runner.create_notebook([
            "from math import pi, sqrt, ceil, floor",
            textwrap.dedent("""\
                print(f"pi={pi:.4f} sqrt16={sqrt(16)} ceil={ceil(3.2)} floor={floor(3.8)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pi=3.1416 sqrt16=4.0 ceil=4 floor=3" in nb_runner.get_output(2)

    def test_nested_package_import(self, nb_runner):
        """Import from nested packages."""
        nb_runner.create_notebook([
            "from os.path import join, basename, dirname",
            textwrap.dedent("""\
                p = join('/home', 'user', 'file.txt')
                print(f"base={basename(p)} dir={dirname(p)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "base=file.txt" in output

    def test_reimport_after_change(self, nb_runner, tmp_path):
        """Module reimported after source change."""
        mod_file = tmp_path / "mymod.py"
        mod_file.write_text("VALUE = 100\n")
        sys_path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{sys_path_str}')",
            "import mymod",
            "print(mymod.VALUE)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

        # Change module and restart
        mod_file.write_text("VALUE = 999\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "999" in nb_runner.get_output(3)


class TestMultiModuleImportInteraction:
    """Test interactions between multiple imported modules."""

    def test_two_modules_interact(self, nb_runner, tmp_path):
        """Two custom modules interact across cells."""
        (tmp_path / "mod_a.py").write_text("def double(x): return x * 2\n")
        (tmp_path / "mod_b.py").write_text(
            "def format_result(val): return f'Result: {val}'\n"
        )
        sys_path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{sys_path_str}')",
            "import mod_a\nimport mod_b",
            textwrap.dedent("""\
                val = mod_a.double(21)
                msg = mod_b.format_result(val)
                print(msg)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Result: 42" in nb_runner.get_output(3)

    def test_import_in_function(self, nb_runner):
        """Import inside function body."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def compute_hash(data):
                    import hashlib
                    return hashlib.md5(data.encode()).hexdigest()[:8]
            """),
            textwrap.dedent("""\
                h = compute_hash("hello world")
                print(f"hash={h}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "hash=" in output
        assert len(nb_runner.get_output(2).split("hash=")[1].strip()) == 8
