"""Batch 57: Multi-file module system — complex module interdependencies with cash."""
import textwrap
import pytest


@pytest.mark.stress
class TestMultiModuleBasics:
    """Test multi-file module systems."""

    def test_module_chain_import(self, nb_runner, tmp_path):
        """Module A imports B, B imports C — chain dependency."""
        pkg = tmp_path / "chain_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod_c.py").write_text("BASE = 10\ndef get_base(): return BASE\n")
        (pkg / "mod_b.py").write_text("from chain_pkg.mod_c import get_base\ndef double(): return get_base() * 2\n")
        (pkg / "mod_a.py").write_text("from chain_pkg.mod_b import double\ndef compute(): return double() + 1\n")
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
            textwrap.dedent("""\
                from chain_pkg.mod_a import compute
                result = compute()
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=21" in nb_runner.get_output(2)  # 10*2+1

    def test_package_with_init(self, nb_runner, tmp_path):
        """Package with __init__.py exporting symbols."""
        pkg = tmp_path / "mathlib"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            "from mathlib.ops import add, multiply\n__version__ = '1.0'\n"
        )
        (pkg / "ops.py").write_text(
            "def add(a, b): return a + b\ndef multiply(a, b): return a * b\n"
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=7 r2=30 ver=1.0" in nb_runner.get_output(2)

    def test_subpackage_imports(self, nb_runner, tmp_path):
        """Nested subpackages with cross-imports."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "__init__.py").write_text("")
        
        utils = root / "utils"
        utils.mkdir()
        (utils / "__init__.py").write_text("")
        (utils / "helpers.py").write_text("def fmt(x): return f'[{x}]'\n")
        
        core = root / "core"
        core.mkdir()
        (core / "__init__.py").write_text("")
        (core / "engine.py").write_text(
            "from project.utils.helpers import fmt\n"
            "def process(data): return [fmt(d) for d in data]\n"
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
            textwrap.dedent("""\
                from project.core.engine import process
                result = process([1, 2, 3])
                print(f"result={result}")
            """),
        ])
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
        (pkg / "__init__.py").write_text("")
        (pkg / "base.py").write_text("FACTOR = 2\n")
        (pkg / "middle.py").write_text(
            "from deep_pkg.base import FACTOR\n"
            "def scaled(x): return x * FACTOR\n"
        )
        (pkg / "top.py").write_text(
            "from deep_pkg.middle import scaled\n"
            "def compute(x): return scaled(x) + 1\n"
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{pkg_parent}')
            """),
            textwrap.dedent("""\
                from deep_pkg.top import compute
                result = compute(5)
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(2)  # 5*2+1

    def test_conditional_import(self, nb_runner, tmp_path):
        """Conditional import based on config."""
        pkg = tmp_path / "cond_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "fast.py").write_text("def process(x): return x * 10\n")
        (pkg / "slow.py").write_text("def process(x): return x + 1\n")
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(2)

    def test_relative_like_imports(self, nb_runner, tmp_path):
        """Simulate relative imports with explicit paths."""
        pkg = tmp_path / "rel_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "constants.py").write_text("PI = 3.14159\nE = 2.71828\n")
        (pkg / "math_ops.py").write_text(
            "from rel_pkg.constants import PI, E\n"
            "def circle_area(r): return PI * r * r\n"
            "def exp_approx(x): return E ** x\n"
        )
        pkg_parent = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "area=78.540" in out
        assert "exp=" in out
