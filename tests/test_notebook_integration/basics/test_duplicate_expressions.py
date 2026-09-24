"""Identical statements and duplicated code in a notebook are cached apart."""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.core]


# Two cells with identical content that the cell id cannot tell apart raise
# RuntimeError, so these tests either vary the cells (a comment is enough) or
# assert the ambiguity error.
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestDuplicateStatements:
    """Same or similar statements in multiple cells."""

    def test_identical_cells_raise_ambiguity(self, nb_runner):
        """Two cells with identical content should raise RuntimeError."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "x = 10",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="Ambiguous cell"):
            nb_runner.run_all()

    def test_similar_code_edit_one(self, nb_runner):
        """Two similar cells (unique comments), edit one."""
        nb_runner.create_notebook(
            [
                "x = 10  # cell A",
                "x = 10  # cell B (overrides A)",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "x = 20  # cell B (overrides A)")
        nb_runner.run_all()
        assert "y = 21" in nb_runner.get_output(3)

    def test_same_computation_different_inputs(self, nb_runner):
        """Same code pattern but different inputs."""
        nb_runner.create_notebook(
            [
                "a = 10\nb = 20",
                "result_a = a * 2\nresult_b = b * 2",
                "print(f'ra = {result_a}, rb = {result_b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ra = 20, rb = 40" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "a = 100\nb = 200")
        nb_runner.run_all()
        assert "ra = 200, rb = 400" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestSimilarFunctions:
    """Similar function definitions in different cells."""

    def test_two_similar_functions(self, nb_runner):
        """Two functions with same structure but different names."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "def triple(x):\n    return x * 3",
                "a = double(5)\nb = triple(5)\nprint(f'a = {a}, b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10, b = 15" in nb_runner.get_output(3)

        # Edit one function
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 20")
        nb_runner.run_all()
        assert "a = 100, b = 15" in nb_runner.get_output(3)

    def test_same_function_name_redefined(self, nb_runner):
        """Same function name defined twice — last definition wins."""
        nb_runner.create_notebook(
            [
                "def f(x):\n    return x + 1",
                "def f(x):\n    return x + 2",
                "result = f(10)\nprint(f'result = {result}')",
            ]
        )
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


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestDuplicateImports:
    """Same import in multiple cells."""

    def test_identical_import_raises_ambiguity(self, nb_runner):
        """Two identical 'import math' cells trigger ambiguity error."""
        nb_runner.create_notebook(
            [
                "import math",
                "import math",
                "val = math.sqrt(16)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="Ambiguous cell"):
            nb_runner.run_all()


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestRepetitivePatterns:
    """Repetitive code patterns across cells."""

    def test_sequential_increments(self, nb_runner):
        """Three cells each incrementing a counter — using unique code."""
        nb_runner.create_notebook(
            [
                "count = 0",
                "count = count + 1  # first increment",
                "count = count + 1  # second increment",
                "count = count + 1  # third increment",
                "print(f'count = {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 3" in nb_runner.get_output(5)

        # Edit middle increment
        nb_runner.set_cell_source(3, "count = count + 10  # second increment (boosted)")
        nb_runner.run_all()
        assert "count = 12" in nb_runner.get_output(5)

    def test_identical_increments_raise_ambiguity(self, nb_runner):
        """Three identical increment cells should raise ambiguity."""
        nb_runner.create_notebook(
            [
                "count = 0",
                "count = count + 1",
                "count = count + 1",
                "count = count + 1",
                "print(f'count = {count}')",
            ]
        )
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="Ambiguous cell"):
            nb_runner.run_all()

    def test_parallel_computations(self, nb_runner):
        """Same pattern applied to different data in parallel cells."""
        nb_runner.create_notebook(
            [
                "data_a = [1, 2, 3]\ndata_b = [10, 20, 30]",
                "sum_a = sum(data_a)\nsum_b = sum(data_b)",
                "print(f'sum_a = {sum_a}, sum_b = {sum_b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum_a = 6, sum_b = 60" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data_a = [10, 20, 30]\ndata_b = [1, 2, 3]")
        nb_runner.run_all()
        assert "sum_a = 60, sum_b = 6" in nb_runner.get_output(3)


# The same expression statement several times in one cell (``c.increment()``
# three times) must run every time. Identical statements differ only in their
# occurrence index in the cell, which is part of the cache key; without it the
# later calls would be restored from the first, and a stateful method would
# change its object once instead of three times.
class TestDuplicateExpressionStatements:
    """Tests for duplicate expression statement deduplication bug."""

    def test_duplicate_method_calls_with_mutation(self, nb_runner):
        """
        Core bug: calling obj.method() multiple times without assignment
        should execute each call, not cache-deduplicate them.
        """
        nb_runner.create_notebook(
            [
                (
                    "class Counter:\n"
                    "    def __init__(self, start=0):\n"
                    "        self.value = start\n"
                    "    def increment(self):\n"
                    "        self.value += 1\n"
                    "        return self.value\n"
                    "\n"
                    "c = Counter(10)\n"
                    "c.increment()\n"
                    "c.increment()\n"
                    "c.increment()\n"
                    "print(f'c.value = {c.value}')"
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "c.value = 13" in out, (
            f"Counter should be 13 (10 + 3 increments). If it shows 11, duplicate calls were cached. Got: {out}"
        )

    def test_duplicate_method_calls_return_values_differ(self, nb_runner):
        """
        When return values are assigned to variables, each call should
        produce different results. This tests the workaround of using
        assignment to differentiate calls.
        """
        nb_runner.create_notebook(
            [
                (
                    "class Counter:\n"
                    "    def __init__(self, start=0):\n"
                    "        self.value = start\n"
                    "    def increment(self):\n"
                    "        self.value += 1\n"
                    "        return self.value\n"
                    "\n"
                    "c = Counter(10)\n"
                    "r1 = c.increment()\n"
                    "r2 = c.increment()\n"
                    "r3 = c.increment()\n"
                    "print(f'results: {r1}, {r2}, {r3}')"
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "results: 11, 12, 13" in out, (
            f"With variable assignment, each call should produce unique results. Got: {out}"
        )

    def test_duplicate_list_append_calls(self, nb_runner):
        """
        Multiple list.append() calls should all execute.
        This is important for accumulator patterns.
        """
        nb_runner.create_notebook(
            [
                ("items = []\nitems.append('a')\nitems.append('b')\nitems.append('c')\nprint(f'items = {items}')"),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "items = ['a', 'b', 'c']" in out, f"All three append calls should execute. Got: {out}"

    def test_duplicate_print_calls_output_correct(self, nb_runner):
        """
        Even though duplicate print calls share cache keys,
        the output should be reproduced correctly on cache restore.
        """
        nb_runner.create_notebook(
            [
                ("print('hello')\nprint('hello')\nprint('hello')"),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        # Count occurrences of 'hello' in the non-badge output
        hello_count = out.count("hello")
        assert hello_count >= 3, (
            f"Three print('hello') calls should produce at least 3 'hello' outputs. Got {hello_count} in: {out}"
        )

    def test_duplicate_calls_across_re_execution(self, nb_runner):
        """
        After initial run, re-executing should still produce correct results
        for duplicate method calls.
        """
        nb_runner.create_notebook(
            [
                (
                    "class Accumulator:\n"
                    "    def __init__(self):\n"
                    "        self.total = 0\n"
                    "    def add(self, val):\n"
                    "        self.total += val\n"
                    "        return self.total\n"
                    "\n"
                    "acc = Accumulator()\n"
                    "acc.add(10)\n"
                    "acc.add(20)\n"
                    "acc.add(30)\n"
                    "print(f'total = {acc.total}')"
                ),
            ]
        )
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "total = 60" in out1, f"First run: expected total=60. Got: {out1}"

        # Second run (should use cache but still be correct)
        nb_runner.run_all()
        out2 = nb_runner.get_output(1)
        assert "total = 60" in out2, f"Second run: expected total=60 (cached). Got: {out2}"

    def test_duplicate_global_function_calls(self, nb_runner):
        """
        Duplicate calls to a global function with side effects should
        each be executed.
        """
        nb_runner.create_notebook(
            [
                (
                    "call_log = []\n"
                    "def track():\n"
                    "    call_log.append(len(call_log))\n"
                    "\n"
                    "track()\n"
                    "track()\n"
                    "track()\n"
                    "print(f'call_log = {call_log}')"
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "call_log = [0, 1, 2]" in out, f"Three track() calls should produce [0, 1, 2]. Got: {out}"
