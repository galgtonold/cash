"""Batch 119 – Duplicate/similar code + cell interaction tests.

Tests that exercise identical or similar code patterns in multiple cells,
testing occurrence_index, cache key disambiguation, and the ambiguity
detection mechanism.

NOTE: Cash raises RuntimeError when two cells have *identical* content
and cannot be resolved by cell ID. Tests here either:
  (a) use unique comments/variations to differentiate, or
  (b) explicitly test the ambiguity error.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestDuplicateStatements:
    """Same or similar statements in multiple cells."""

    def test_identical_cells_raise_ambiguity(self, nb_runner):
        """Two cells with identical content should raise RuntimeError."""
        nb_runner.create_notebook([
            "x = 10",
            "x = 10",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="Ambiguous cell"):
            nb_runner.run_all()

    def test_similar_assignment_with_comments(self, nb_runner):
        """Two cells with same logic but unique comments."""
        nb_runner.create_notebook([
            "x = 10  # first assignment",
            "x = 10  # second assignment (override)",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 10" in nb_runner.get_output(3)

    def test_similar_code_edit_one(self, nb_runner):
        """Two similar cells (unique comments), edit one."""
        nb_runner.create_notebook([
            "x = 10  # cell A",
            "x = 10  # cell B (overrides A)",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "x = 20  # cell B (overrides A)")
        nb_runner.run_all()
        assert "y = 21" in nb_runner.get_output(3)

    def test_same_computation_different_inputs(self, nb_runner):
        """Same code pattern but different inputs."""
        nb_runner.create_notebook([
            "a = 10\nb = 20",
            "result_a = a * 2\nresult_b = b * 2",
            "print(f'ra = {result_a}, rb = {result_b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ra = 20, rb = 40" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "a = 100\nb = 200")
        nb_runner.run_all()
        assert "ra = 200, rb = 400" in nb_runner.get_output(3)


class TestSimilarFunctions:
    """Similar function definitions in different cells."""

    def test_two_similar_functions(self, nb_runner):
        """Two functions with same structure but different names."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2",
            "def triple(x):\n    return x * 3",
            "a = double(5)\nb = triple(5)\nprint(f'a = {a}, b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10, b = 15" in nb_runner.get_output(3)

        # Edit one function
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 20")
        nb_runner.run_all()
        assert "a = 100, b = 15" in nb_runner.get_output(3)

    def test_same_function_name_redefined(self, nb_runner):
        """Same function name defined twice — last definition wins."""
        nb_runner.create_notebook([
            "def f(x):\n    return x + 1",
            "def f(x):\n    return x + 2",
            "result = f(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)

        # Edit first definition — shouldn't matter since second overrides
        nb_runner.set_cell_source(1, "def f(x):\n    return x + 100")
        nb_runner.run_all()
        # Second cell still defines f as x + 2
        assert "result = 12" in nb_runner.get_output(3)

        # Now edit the second (winning) definition
        nb_runner.set_cell_source(2, "def f(x):\n    return x * 10")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(3)


class TestDuplicateImports:
    """Same import in multiple cells."""

    def test_identical_import_raises_ambiguity(self, nb_runner):
        """Two identical 'import math' cells trigger ambiguity error."""
        nb_runner.create_notebook([
            "import math",
            "import math",
            "val = math.sqrt(16)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="Ambiguous cell"):
            nb_runner.run_all()

    def test_import_with_unique_usage(self, nb_runner):
        """Import in two cells with unique additional code — no ambiguity."""
        nb_runner.create_notebook([
            "import math  # primary import",
            "import math  # secondary import (just in case)",
            "val = math.sqrt(16)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 4.0" in nb_runner.get_output(3)

    def test_import_then_from_import(self, nb_runner):
        """import X then from X import Y."""
        nb_runner.create_notebook([
            "import math",
            "from math import pi",
            "val = math.sqrt(pi)\nprint(f'val = {val:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 1.7725" in nb_runner.get_output(3)


class TestRepetitivePatterns:
    """Repetitive code patterns across cells."""

    def test_sequential_increments(self, nb_runner):
        """Three cells each incrementing a counter — using unique code."""
        nb_runner.create_notebook([
            "count = 0",
            "count = count + 1  # first increment",
            "count = count + 1  # second increment",
            "count = count + 1  # third increment",
            "print(f'count = {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 3" in nb_runner.get_output(5)

        # Edit middle increment
        nb_runner.set_cell_source(3, "count = count + 10  # second increment (boosted)")
        nb_runner.run_all()
        assert "count = 12" in nb_runner.get_output(5)

    def test_identical_increments_raise_ambiguity(self, nb_runner):
        """Three identical increment cells should raise ambiguity."""
        nb_runner.create_notebook([
            "count = 0",
            "count = count + 1",
            "count = count + 1",
            "count = count + 1",
            "print(f'count = {count}')",
        ])
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="Ambiguous cell"):
            nb_runner.run_all()

    def test_parallel_computations(self, nb_runner):
        """Same pattern applied to different data in parallel cells."""
        nb_runner.create_notebook([
            "data_a = [1, 2, 3]\ndata_b = [10, 20, 30]",
            "sum_a = sum(data_a)\nsum_b = sum(data_b)",
            "print(f'sum_a = {sum_a}, sum_b = {sum_b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum_a = 6, sum_b = 60" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data_a = [10, 20, 30]\ndata_b = [1, 2, 3]")
        nb_runner.run_all()
        assert "sum_a = 60, sum_b = 6" in nb_runner.get_output(3)
