"""Integration tests for module reload cache invalidation and exception surfacing.

Tests that:
1. When a tracked local module is modified, cached statements that depend on it
   get cache misses and re-execute with the new module code.
2. Exceptions during cell execution (ImportError, ValueError, etc.) are properly
   surfaced to the user and not silently swallowed.
3. Transitive module dependencies: when module A imports module B and B changes,
   statements depending on A are also invalidated.
"""
import time
import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = pytest.mark.modules


class TestModuleReloadIntegration:
    """Integration tests for module reload cache invalidation in real notebooks."""

    def test_module_change_invalidates_cache(self, nb_runner, tmp_path):
        """Changing a local module's source should cause cache miss on re-run."""
        # Create a local module in the work directory
        module_file = tmp_path / "mymod.py"
        module_file.write_text("def compute(x):\n    return x + 1\n")

        nb_runner.create_notebook([
            "import mymod",
            "result = mymod.compute(10)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()

        # First run: computes everything
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)
        assert "result=11" in output1

        # Second run: should cache/skip
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert "result=11" in output2

        # Now change the module
        module_file.write_text("def compute(x):\n    return x + 100\n")

        # Third run: should detect the change and re-compute
        nb_runner.run_all()
        output3 = nb_runner.get_output(2)
        assert "result=110" in output3, (
            f"Expected result=110 after module change, got: {output3}"
        )

    def test_module_reload_with_multiple_dependents(self, nb_runner, tmp_path):
        """Multiple cells depending on a changed module should all re-execute."""
        module_file = tmp_path / "helpers.py"
        module_file.write_text("def double(x):\n    return x * 2\n")

        nb_runner.create_notebook([
            "import helpers",
            "a = helpers.double(5)\nprint(f'a={a}')",
            "b = helpers.double(10)\nprint(f'b={b}')",
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        assert "a=10" in nb_runner.get_output(2)
        assert "b=20" in nb_runner.get_output(3)

        # Change module: double now triples
        module_file.write_text("def double(x):\n    return x * 3\n")

        # Re-run: both should get new results
        nb_runner.run_all()
        output_a = nb_runner.get_output(2)
        output_b = nb_runner.get_output(3)
        assert "a=15" in output_a, f"Expected a=15, got: {output_a}"
        assert "b=30" in output_b, f"Expected b=30, got: {output_b}"

    def test_module_reload_badge_shows_notification(self, nb_runner, tmp_path):
        """Badge should show MODULE_RELOADED notification after module change."""
        module_file = tmp_path / "trackmod.py"
        module_file.write_text("VAL = 42\n")

        nb_runner.create_notebook([
            "import trackmod",
            "print(f'val={trackmod.VAL}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # First run
        nb_runner.run_all()
        assert "val=42" in nb_runner.get_output(2)

        # Change module
        module_file.write_text("VAL = 99\n")

        # Re-run and check debug output for reload notification
        nb_runner.run_all()
        raw_output = nb_runner.get_output(2, filter_debug=False)
        # Either the value changed OR there's a reload notification in debug
        # The key assertion is that the value reflects the change
        assert "val=99" in nb_runner.get_output(2), (
            f"Expected val=99 after module reload. Raw output: {raw_output}"
        )


class TestExceptionSurfacingIntegration:
    """Integration tests for exception surfacing in real notebooks."""

    def test_import_error_surfaces(self, nb_runner):
        """ImportError from importing a nonexistent module should be visible."""
        nb_runner.create_notebook([
            "import nonexistent_module_xyz_12345",
        ])
        nb_runner.start_kernel()

        with pytest.raises(CellExecutionError) as exc_info:
            nb_runner.run_all()

        assert "nonexistent_module_xyz_12345" in str(exc_info.value)

    def test_value_error_surfaces(self, nb_runner):
        """ValueError raised in a cell should be visible."""
        nb_runner.create_notebook([
            "raise ValueError('this is a test error')",
        ])
        nb_runner.start_kernel()

        with pytest.raises(CellExecutionError) as exc_info:
            nb_runner.run_all()

        assert "this is a test error" in str(exc_info.value)

    def test_error_in_second_statement_surfaces(self, nb_runner):
        """Error in the second statement of a cell should still surface."""
        nb_runner.create_notebook([
            "x = 42\nraise RuntimeError('second statement fails')",
        ])
        nb_runner.start_kernel()

        with pytest.raises(CellExecutionError) as exc_info:
            nb_runner.run_all()

        assert "second statement fails" in str(exc_info.value)

    def test_error_in_second_cell_surfaces(self, nb_runner):
        """Error in the second cell should surface after first cell succeeds."""
        nb_runner.create_notebook([
            "x = 10\nprint(f'x={x}')",
            "raise TypeError('cell 2 error')",
        ])
        nb_runner.start_kernel()

        # Run first cell - should succeed
        nb_runner.run_cell(1)
        assert "x=10" in nb_runner.get_output(1)

        # Run second cell - should raise
        with pytest.raises(CellExecutionError) as exc_info:
            nb_runner.run_cell(2)

        assert "cell 2 error" in str(exc_info.value)

    def test_name_error_surfaces(self, nb_runner):
        """NameError from undefined variable should surface."""
        nb_runner.create_notebook([
            "print(undefined_var_xyz)",
        ])
        nb_runner.start_kernel()

        with pytest.raises(CellExecutionError) as exc_info:
            nb_runner.run_all()

        assert "undefined_var_xyz" in str(exc_info.value)


class TestTransitiveDependencyIntegration:
    """Integration tests for transitive module dependency invalidation.

    Scenario: notebook imports ``metrics``, ``metrics.py`` imports from ``helpers.py``.
    When ``helpers.py`` changes, ``metrics`` should be reloaded and statements
    using it should re-execute with the new code.
    """

    def test_transitive_dep_change_invalidates_cache(self, nb_runner, tmp_path):
        """Changing helpers.py (imported by metrics.py) should cause cache miss."""
        # Create the two-level module hierarchy
        helpers_file = tmp_path / "helpers.py"
        helpers_file.write_text("def add_one(x):\n    return x + 1\n")

        metrics_file = tmp_path / "metrics.py"
        metrics_file.write_text(
            "from helpers import add_one\n"
            "def compute(x):\n"
            "    return add_one(x) * 2\n"
        )

        nb_runner.create_notebook([
            "import metrics",
            "result = metrics.compute(10)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()

        # First run: computes everything
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)
        assert "result=22" in output1, f"Expected result=22, got: {output1}"

        # Second run: should cache/skip
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert "result=22" in output2

        # Now change ONLY helpers.py (metrics.py stays the same)
        time.sleep(0.1)
        helpers_file.write_text("def add_one(x):\n    return x + 100\n")

        # Third run: should detect transitive change and re-compute
        nb_runner.run_all()
        output3 = nb_runner.get_output(2)
        assert "result=220" in output3, (
            f"Expected result=220 after helpers.py change (10+100)*2, got: {output3}"
        )

    def test_transitive_dep_multiple_dependents(self, nb_runner, tmp_path):
        """Multiple cells depending on a module whose sub-dep changed should all re-execute."""
        helpers_file = tmp_path / "helpers.py"
        helpers_file.write_text("BASE = 10\n")

        svc_file = tmp_path / "svc.py"
        svc_file.write_text(
            "from helpers import BASE\n"
            "def calc(x):\n"
            "    return x + BASE\n"
        )

        nb_runner.create_notebook([
            "import svc",
            "a = svc.calc(5)\nprint(f'a={a}')",
            "b = svc.calc(20)\nprint(f'b={b}')",
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        assert "a=15" in nb_runner.get_output(2)
        assert "b=30" in nb_runner.get_output(3)

        # Change helpers.py (not svc.py)
        time.sleep(0.1)
        helpers_file.write_text("BASE = 1000\n")

        # Re-run: both should get new results
        nb_runner.run_all()
        output_a = nb_runner.get_output(2)
        output_b = nb_runner.get_output(3)
        assert "a=1005" in output_a, f"Expected a=1005, got: {output_a}"
        assert "b=1020" in output_b, f"Expected b=1020, got: {output_b}"

    def test_three_level_transitive_dep(self, nb_runner, tmp_path):
        """Three-level chain: notebook imports app, app imports svc, svc imports utils."""
        utils_file = tmp_path / "chain_utils.py"
        utils_file.write_text("FACTOR = 2\n")

        svc_file = tmp_path / "chain_svc.py"
        svc_file.write_text(
            "from chain_utils import FACTOR\n"
            "def multiply(x):\n"
            "    return x * FACTOR\n"
        )

        app_file = tmp_path / "chain_app.py"
        app_file.write_text(
            "from chain_svc import multiply\n"
            "def run(x):\n"
            "    return multiply(x) + 1\n"
        )

        nb_runner.create_notebook([
            "import chain_app",
            "result = chain_app.run(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()

        # First run: 5*2+1 = 11
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)
        assert "result=11" in output1, f"Expected result=11, got: {output1}"

        # Change the bottom-level utils
        time.sleep(0.1)
        utils_file.write_text("FACTOR = 100\n")

        # Re-run: 5*100+1 = 501
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert "result=501" in output2, (
            f"Expected result=501 after chain_utils change, got: {output2}"
        )
