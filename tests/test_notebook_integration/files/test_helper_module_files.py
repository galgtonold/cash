"""Helper module files written from the notebook and edited on disk."""

import time

import pytest


@pytest.mark.files
class TestModuleHotReload:
    """Test module hot-reload detection."""

    def test_external_module_function_change(self, nb_runner, tmp_path):
        """
        Import a module, change it on disk, and verify that cash detects
        the change and recomputes on run_all().
        """
        mod_path = tmp_path / "mymodule.py"
        mod_path.write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "import mymodule",
                "result = mymodule.compute(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 20" in nb_runner.get_output(2)

        # Modify the module on disk
        mod_path.write_text("def compute(x):\n    return x * 3\n", encoding="utf-8")
        time.sleep(0.5)

        # Re-run all - module reload should detect change
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 30" in out, f"Expected 30 after module change, got: {out}"

    def test_module_class_method_change(self, nb_runner, tmp_path):
        """
        Import a class from a module, change a method, verify detection.
        """
        mod_path = tmp_path / "shapes.py"
        mod_path.write_text(
            "class Circle:\n"
            "    def __init__(self, r):\n"
            "        self.r = r\n"
            "    def area(self):\n"
            "        return 3.14 * self.r ** 2\n",
            encoding="utf-8",
        )

        nb_runner.create_notebook(
            [
                "import shapes",
                "c = shapes.Circle(5)\nprint(f'area = {c.area()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "area = 78.5" in out, f"Got: {out}"

        # Change pi approximation
        mod_path.write_text(
            "class Circle:\n"
            "    def __init__(self, r):\n"
            "        self.r = r\n"
            "    def area(self):\n"
            "        import math\n"
            "        return math.pi * self.r ** 2\n",
            encoding="utf-8",
        )
        time.sleep(0.5)

        # Re-run all - module change should be detected
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "area = 78.5398" in out2, f"Expected math.pi result, got: {out2}"

    def test_module_new_function_added(self, nb_runner, tmp_path):
        """
        Add a new function to an existing module and use it.
        """
        mod_path = tmp_path / "utils.py"
        mod_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "import utils",
                "r1 = utils.add(3, 4)\nprint(f'r1 = {r1}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "r1 = 7" in nb_runner.get_output(2)

        # Add a multiply function to the module
        mod_path.write_text(
            "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n", encoding="utf-8"
        )
        time.sleep(0.5)

        # Modify cell 2 to also use the new function
        nb_runner.set_cell_source(2, "r1 = utils.add(3, 4)\nr2 = utils.multiply(3, 4)\nprint(f'r1={r1}, r2={r2}')")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "r1=7" in out, f"Got: {out}"
        assert "r2=12" in out, f"Expected r2=12, got: {out}"

    def test_from_import_module_reload(self, nb_runner, tmp_path):
        """
        Test that 'from X import Y' style imports detect module changes.
        When a module is modified, re-running the import should pick up the
        new function definition and downstream cells should use the updated version.

        This tests the F3 bug fix: previously from-import style wouldn't detect
        module changes because (1) the redundant import optimization skipped the
        re-import, (2) tracking for from-imported names wasn't cleared, and
        (3) the cache key for import statements didn't include module source hash.
        """
        mod_path = tmp_path / "mathlib.py"
        mod_path.write_text("def square(x):\n    return x ** 2\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "from mathlib import square",
                "result = square(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 25" in nb_runner.get_output(2)

        # Modify the module to cube instead
        mod_path.write_text("def square(x):\n    return x ** 3\n", encoding="utf-8")
        time.sleep(0.5)

        # Re-run all — cash should detect the module change and re-execute
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 125" in out, f"from-import module reload failed: expected 125 (5**3), got: {out}"
