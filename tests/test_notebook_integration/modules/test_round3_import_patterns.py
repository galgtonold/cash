"""
Import patterns — dynamic imports, conditional imports, importlib,
sys.path manipulation, and star imports across cells.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.modules]


class TestDynamicImportPatterns:
    """Test caching with dynamic import patterns."""

    def test_reimport_after_change(self, nb_runner, tmp_path):
        """Module reimported after source change."""
        mod_file = tmp_path / "mymod.py"
        mod_file.write_text("VALUE = 100\n")
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
        (tmp_path / "mod_b.py").write_text("def format_result(val): return f'Result: {val}'\n")
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
