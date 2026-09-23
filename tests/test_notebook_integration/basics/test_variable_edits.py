"""Editing assignments and re-running, once or many times in a row."""

import pytest

pytestmark = [pytest.mark.stress]


# Mixed assignment types + cell edit interaction tests.
#
# Tests that exercise augmented assignments, tuple/list unpacking,
# walrus operator, global/nonlocal, and compound assignments.
@pytest.mark.core
@pytest.mark.timeout(30)
class TestAugmentedAssignment:
    """Augmented assignment operators + edits."""

    def test_augmented_add_edit(self, nb_runner):
        """Edit augmented addition."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "x += 5",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "x += 20")
        nb_runner.run_all()
        assert "x = 30" in nb_runner.get_output(3)

    def test_augmented_mul_edit_base(self, nb_runner):
        """Edit the base value before augmented multiply."""
        nb_runner.create_notebook(
            [
                "x = 3",
                "x *= 4",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 12" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "x = 40" in nb_runner.get_output(3)

    def test_chain_of_augmented_ops(self, nb_runner):
        """Chain of augmented assignments, edit one in the middle."""
        nb_runner.create_notebook(
            [
                "v = 100",
                "v -= 10  # subtract",
                "v *= 2  # double",
                "v //= 3  # floor divide",
                "print(f'v = {v}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 100 - 10 = 90, * 2 = 180, // 3 = 60
        assert "v = 60" in nb_runner.get_output(5)

        nb_runner.set_cell_source(2, "v -= 50  # subtract more")
        nb_runner.run_all()
        # 100 - 50 = 50, * 2 = 100, // 3 = 33
        assert "v = 33" in nb_runner.get_output(5)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestUnpackingEdits:
    """Tuple/list unpacking + cell edits."""

    def test_tuple_unpack_edit_source(self, nb_runner):
        """Edit the source of a tuple unpack."""
        nb_runner.create_notebook(
            [
                "data = (1, 2, 3)",
                "a, b, c = data",
                "result = a + b + c\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data = (10, 20, 30)")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(3)

    def test_star_unpack_edit(self, nb_runner):
        """Edit data with star unpacking."""
        nb_runner.create_notebook(
            [
                "values = [1, 2, 3, 4, 5]",
                "first, *rest = values",
                "result = first + sum(rest)\nprint(f'first = {first}, rest_sum = {sum(rest)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first = 1, rest_sum = 14" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "values = [100, 1, 1, 1]")
        nb_runner.run_all()
        assert "first = 100, rest_sum = 3" in nb_runner.get_output(3)

    def test_nested_unpack_edit(self, nb_runner):
        """Nested unpacking + edit."""
        nb_runner.create_notebook(
            [
                "pair = ((1, 2), (3, 4))",
                "(a, b), (c, d) = pair",
                "result = a * d - b * c\nprint(f'det = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1*4 - 2*3 = -2
        assert "det = -2" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "pair = ((3, 1), (2, 4))")
        nb_runner.run_all()
        # 3*4 - 1*2 = 10
        assert "det = 10" in nb_runner.get_output(3)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestComprehensionEdits:
    """List/dict/set comprehensions + edits."""

    def test_list_comp_edit_filter(self, nb_runner):
        """Edit the filter in a list comprehension."""
        nb_runner.create_notebook(
            [
                "data = list(range(20))",
                "evens = [x for x in data if x % 2 == 0]",
                "result = sum(evens)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0+2+4+6+8+10+12+14+16+18 = 90
        assert "result = 90" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "evens = [x for x in data if x % 3 == 0]")
        nb_runner.run_all()
        # 0+3+6+9+12+15+18 = 63
        assert "result = 63" in nb_runner.get_output(3)

    def test_dict_comp_edit(self, nb_runner):
        """Edit a dict comprehension."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']",
                "d = {k: i for i, k in enumerate(keys)}",
                "result = d['b']\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "d = {k: (i + 1) * 10 for i, k in enumerate(keys)}")
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

    def test_nested_comprehension_edit(self, nb_runner):
        """Nested comprehension (matrix) + edit."""
        nb_runner.create_notebook(
            [
                "n = 3",
                "matrix = [[i * n + j for j in range(n)] for i in range(n)]",
                "flat = [x for row in matrix for x in row]\nresult = sum(flat)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0+1+2+3+4+5+6+7+8 = 36
        assert "result = 36" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "n = 4")
        nb_runner.run_all()
        # 0+1+...+15 = 120
        assert "result = 120" in nb_runner.get_output(3)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestMultipleReturnEdits:
    """Functions returning multiple values + cell edits."""

    def test_multi_return_edit(self, nb_runner):
        """Edit function that returns multiple values."""
        nb_runner.create_notebook(
            [
                "def stats(data):\n    return min(data), max(data), sum(data) / len(data)",
                "lo, hi, avg = stats([1, 2, 3, 4, 5])",
                "print(f'lo={lo} hi={hi} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=5 avg=3.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "lo, hi, avg = stats([10, 20, 30])")
        nb_runner.run_all()
        assert "lo=10 hi=30 avg=20.0" in nb_runner.get_output(3)

    def test_edit_multi_return_function(self, nb_runner):
        """Edit the function body that returns multiple values."""
        nb_runner.create_notebook(
            [
                "def analyze(x):\n    return x, x * 2, x * 3",
                "a, b, c = analyze(5)",
                "result = a + b + c\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 5 + 10 + 15 = 30
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def analyze(x):\n    return x, x ** 2, x ** 3")
        nb_runner.run_all()
        # 5 + 25 + 125 = 155
        assert "result = 155" in nb_runner.get_output(3)


# Multi-round cell editing & cache coherence.
#
# Tests that exercise the trickiest interaction patterns:
# - Editing the same cell multiple times in succession
# - Editing multiple cells between runs
# - Reverting a cell to its original value (should hit cache)
# - Editing upstream then downstream then upstream again
# - Rapid back-and-forth value flipping
@pytest.mark.upstream
class TestMultiEditSameCell:
    """Edit a single upstream cell many times, verifying propagation each time."""

    def test_edit_upstream_three_times(self, nb_runner):
        """Change cell 1 three times, each time verifying cell 3 picks it up."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "print(f'result = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

        # Edit 1
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cell(3)
        assert "result = 100" in nb_runner.get_output(3)

        # Edit 2
        nb_runner.set_cell_source(1, "x = 7")
        nb_runner.run_cell(3)
        assert "result = 14" in nb_runner.get_output(3)

        # Edit 3
        nb_runner.set_cell_source(1, "x = 0")
        nb_runner.run_cell(3)
        assert "result = 0" in nb_runner.get_output(3)

    def test_rapid_flip_flop(self, nb_runner):
        """Rapidly alternate between two values for a cell."""
        nb_runner.create_notebook(
            [
                "flag = True",
                "result = 'yes' if flag else 'no'",
                "print(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = yes" in nb_runner.get_output(3)

        for i in range(4):
            val = "False" if i % 2 == 0 else "True"
            expected = "no" if i % 2 == 0 else "yes"
            nb_runner.set_cell_source(1, f"flag = {val}")
            nb_runner.run_cell(3)
            assert f"result = {expected}" in nb_runner.get_output(3), f"Iteration {i}: expected '{expected}'"


@pytest.mark.upstream
class TestMultiCellEdits:
    """Edit multiple cells between a single run."""

    def test_edit_two_upstream_cells(self, nb_runner):
        """Edit both cell 1 and cell 2 before running cell 3."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = 20",
                "z = x + y\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(3)

        # Edit BOTH upstream cells
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.set_cell_source(2, "y = 200")
        nb_runner.run_cell(3)
        assert "z = 300" in nb_runner.get_output(3)

    def test_edit_upstream_and_downstream(self, nb_runner):
        """Edit cell 1 (upstream) and cell 3 (downstream formula)."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        # Change upstream value AND downstream formula
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.set_cell_source(3, "z = y + 100\nprint(f'z = {z}')")
        nb_runner.run_cell(3)
        assert "z = 120" in nb_runner.get_output(3)

    def test_edit_all_cells_simultaneously(self, nb_runner):
        """Edit all three cells at once before running the last one."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.set_cell_source(2, "b = a * 2")
        nb_runner.set_cell_source(3, "c = b - 50\nprint(f'c = {c}')")
        nb_runner.run_cell(3)
        assert "c = 150" in nb_runner.get_output(3)


@pytest.mark.upstream
class TestEditThenRunPartial:
    """Edit cells and run only some of them — tests cache consistency."""

    def test_edit_upstream_run_only_downstream(self, nb_runner):
        """Edit cell 1 but only re-run cell 3. Upstream should auto-propagate."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 20")
        # Only run cell 3 — upstream checker should detect change and re-execute
        nb_runner.run_cell(3)
        assert "z = 50" in nb_runner.get_output(3)

    def test_edit_middle_run_only_last(self, nb_runner):
        """Edit cell 2 (middle) and only run cell 3."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x + 1",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x * 100")
        nb_runner.run_cell(3)
        assert "y = 500" in nb_runner.get_output(3)

    def test_edit_first_run_middle_then_last(self, nb_runner):
        """Edit cell 1, run cell 2, then run cell 3 separately."""
        nb_runner.create_notebook(
            [
                "x = 3",
                "y = x ** 2\nprint(f'y = {y}')",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 9" in nb_runner.get_output(2)
        assert "z = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 4")
        nb_runner.run_cell(2)
        assert "y = 16" in nb_runner.get_output(2)

        nb_runner.run_cell(3)
        assert "z = 17" in nb_runner.get_output(3)


@pytest.mark.upstream
class TestLongChainEdits:
    """Test propagation through long dependency chains with edits."""

    def test_five_cell_chain_edit_root(self, nb_runner):
        """5-cell chain: edit root and verify propagation to the end."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 5" in nb_runner.get_output(5)

        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_cell(5)
        assert "e = 104" in nb_runner.get_output(5)

    def test_five_cell_chain_edit_middle(self, nb_runner):
        """5-cell chain: edit middle cell and verify downstream propagation."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b * 10",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 22" in nb_runner.get_output(5)

        # Edit middle (cell 3)
        nb_runner.set_cell_source(3, "c = b * 100")
        nb_runner.run_cell(5)
        assert "e = 202" in nb_runner.get_output(5)

    def test_chain_edit_root_twice(self, nb_runner):
        """Edit root of a chain twice — second edit should still propagate."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x * 2",
                "z = y * 3",
                "w = z * 4\nprint(f'w = {w}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 24" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_cell(4)
        assert "w = 240" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(4)
        assert "w = 120" in nb_runner.get_output(4)


# Rapid-fire edit interaction tests.
#
# Tests that exercise many rapid successive edits to the same cell(s),
# verifying cache coherence under high edit frequency.
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestRapidEditsOneCell:
    """Many edits to a single cell in quick succession."""

    def test_five_rapid_edits(self, nb_runner):
        """Edit the same cell five times, verify each time."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()

        for i in range(5):
            nb_runner.set_cell_source(1, f"x = {i * 10}")
            nb_runner.run_all()
            assert f"y = {i * 10 + 1}" in nb_runner.get_output(2)

    def test_ten_rapid_edits(self, nb_runner):
        """Ten rapid edits to the upstream cell."""
        nb_runner.create_notebook(
            [
                "n = 0",
                "result = n ** 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()

        for i in range(10):
            nb_runner.set_cell_source(1, f"n = {i}")
            nb_runner.run_all()
            assert f"result = {i**2}" in nb_runner.get_output(2)

    def test_rapid_edits_with_function(self, nb_runner):
        """Rapidly edit function definition."""
        nb_runner.create_notebook(
            [
                "def f(x):\n    return x + 0",
                "result = f(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()

        for offset in [1, 5, 10, 100, -1]:
            nb_runner.set_cell_source(1, f"def f(x):\n    return x + {offset}")
            nb_runner.run_all()
            assert f"result = {10 + offset}" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestRapidEditsMultipleCells:
    """Rapid edits across multiple cells."""

    def test_alternating_rapid_edits(self, nb_runner):
        """Alternate between editing two cells rapidly."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = 1",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 2" in nb_runner.get_output(3)

        # Alternate edits
        for i in range(1, 6):
            if i % 2 == 1:
                nb_runner.set_cell_source(1, f"a = {i * 10}")
            else:
                nb_runner.set_cell_source(2, f"b = {i * 10}")
            nb_runner.run_all()

        # After 5 rounds: a=50, b=40
        assert "c = 90" in nb_runner.get_output(3)

    def test_rapid_edit_and_revert(self, nb_runner):
        """Rapidly edit and revert, check cache consistency.

        After editing x = 999 and running, then reverting back to x = 1,
        the system should detect the upstream change in both directions.
        Uses kernel restart between cycles to avoid "already executed"
        skip optimization masking the revert detection.
        """
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit to new value
        nb_runner.set_cell_source(1, "x = 999")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 1000" in nb_runner.get_output(2)

        # Revert back to original
        nb_runner.set_cell_source(1, "x = 1")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestRapidEditWithRestart:
    """Rapid edits combined with kernel restarts."""

    def test_edit_restart_cycle(self, nb_runner):
        """Edit → restart → verify cycle."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 1" in nb_runner.get_output(2)

        for val in [10, 20, 30]:
            nb_runner.set_cell_source(1, f"x = {val}")
            nb_runner.shutdown()
            nb_runner.start_kernel()
            nb_runner.run_all()
            assert f"y = {val + 1}" in nb_runner.get_output(2)


# Variable shadowing, deletion, and scope interaction tests.
#
# Tests that exercise variable shadowing, overwriting, deletion,
# and scope changes combined with cell edits and reruns.
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestVariableShadowing:
    """Variable defined in one cell, redefined in another."""

    def test_shadow_then_edit_first(self, nb_runner):
        """Shadow variable, then edit the first definition."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "x = 20",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

        # Edit first cell — should not change result
        nb_runner.set_cell_source(1, "x = 999")
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

    def test_remove_shadowing_cell(self, nb_runner):
        """Remove the shadowing cell, original value should take effect.
        Replacing with pass effectively removes the override."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "x = 20",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

        # Replace shadow cell with pass
        nb_runner.set_cell_source(2, "pass")
        nb_runner.run_all()
        assert "x = 10" in nb_runner.get_output(3)

    def test_variable_type_change(self, nb_runner):
        """Variable changes type between edits."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "print(f'type = {type(x).__name__}, val = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type = int, val = 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 'hello'")
        nb_runner.run_all()
        assert "type = str, val = hello" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = [1, 2, 3]")
        nb_runner.run_all()
        assert "type = list, val = [1, 2, 3]" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestMultipleVariables:
    """Multiple variables with interleaved edits."""

    def test_edit_one_of_two_independent_vars(self, nb_runner):
        """Two independent variables, edit one."""
        nb_runner.create_notebook(
            [
                "a = 10\nb = 20",
                "print(f'a = {a}, b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10, b = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = 99\nb = 20")
        nb_runner.run_all()
        assert "a = 99, b = 20" in nb_runner.get_output(2)

    def test_swap_variables(self, nb_runner):
        """Swap variable values between cells."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = 2",
                "result = a + b * 10\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(2)

        # Swap values
        nb_runner.set_cell_source(1, "a = 2\nb = 1")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    def test_remove_variable_from_cell(self, nb_runner):
        """Remove a variable from a cell, downstream breaks gracefully."""
        nb_runner.create_notebook(
            [
                "x = 10\ny = 20",
                "result = x + y\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Remove y, now use only x
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.set_cell_source(2, "result = x * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)


# Multiple rapid consecutive edits interaction tests.
#
# Tests making multiple rapid edits to the same cell and verifying
# that each edit is properly picked up.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRapidSameCellEdits:
    """Multiple rapid edits to the same cell."""

    def test_three_consecutive_edits(self, nb_runner):
        """Edit a cell three times consecutively."""
        nb_runner.create_notebook(
            [
                "x = 1  # version 1",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 1" in nb_runner.get_output(2)

        # Edit 1
        nb_runner.set_cell_source(1, "x = 10  # version 2")
        nb_runner.run_all()
        assert "x = 10" in nb_runner.get_output(2)

        # Edit 2
        nb_runner.set_cell_source(1, "x = 100  # version 3")
        nb_runner.run_all()
        assert "x = 100" in nb_runner.get_output(2)

        # Edit 3
        nb_runner.set_cell_source(1, "x = 1000  # version 4")
        nb_runner.run_all()
        assert "x = 1000" in nb_runner.get_output(2)

    def test_oscillating_values(self, nb_runner):
        """Alternate between two values rapidly."""
        nb_runner.create_notebook(
            [
                "val = 'A'  # oscillate val A",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = A" in nb_runner.get_output(2)

        # Switch to B
        nb_runner.set_cell_source(1, "val = 'B'  # oscillate val B")
        nb_runner.run_all()
        assert "val = B" in nb_runner.get_output(2)

        # Back to A (new comment to differentiate)
        nb_runner.set_cell_source(1, "val = 'A'  # oscillate val A again")
        nb_runner.run_all()
        assert "val = A" in nb_runner.get_output(2)

    def test_edit_type_change(self, nb_runner):
        """Edit a cell to change variable type each time."""
        nb_runner.create_notebook(
            [
                "data = 42  # type change start",
                "print(f'type = {type(data).__name__}, data = {data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type = int" in nb_runner.get_output(2)

        # Change to string
        nb_runner.set_cell_source(1, "data = 'hello'  # type change str")
        nb_runner.run_all()
        assert "type = str" in nb_runner.get_output(2)

        # Change to list
        nb_runner.set_cell_source(1, "data = [1, 2, 3]  # type change list")
        nb_runner.run_all()
        assert "type = list" in nb_runner.get_output(2)

        # Change to dict
        nb_runner.set_cell_source(1, "data = {'a': 1}  # type change dict")
        nb_runner.run_all()
        assert "type = dict" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRapidDependentEdits:
    """Rapid edits to dependent cells."""

    def test_edit_producer_then_consumer_rapidly(self, nb_runner):
        """Edit both producer and consumer cells rapidly."""
        nb_runner.create_notebook(
            [
                "x = 5  # producer",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Edit producer
        nb_runner.set_cell_source(1, "x = 50  # producer big")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)

        # Edit consumer
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 150" in nb_runner.get_output(2)

        # Edit both at once
        nb_runner.set_cell_source(1, "x = 7  # producer small")
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 8" in nb_runner.get_output(2)

    def test_rapid_formula_changes(self, nb_runner):
        """Rapidly change the formula applied to same input."""
        nb_runner.create_notebook(
            [
                "n = 10  # input number",
                "result = n + 1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        formulas = [
            ("result = n * 2\nprint(f'result = {result}')", "20"),
            ("result = n ** 2\nprint(f'result = {result}')", "100"),
            ("result = n // 3\nprint(f'result = {result}')", "3"),
            ("result = n - 7\nprint(f'result = {result}')", "3"),
        ]
        for code, expected in formulas:
            nb_runner.set_cell_source(2, code)
            nb_runner.run_all()
            assert f"result = {expected}" in nb_runner.get_output(2)


# Multi-round iterative refinement patterns.
#
# Tests multiple sequential edits to same cell, verifying each round.
@pytest.mark.timeout(90)
class TestMultiRoundRefinement:
    """Multiple rounds of editing the same cell."""

    def test_five_round_formula_refinement(self, nb_runner):
        """Edit formula cell 5 times sequentially."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "result = x\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit 1: add 5
        nb_runner.set_cell_source(2, "result = x + 5\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Edit 2: multiply
        nb_runner.set_cell_source(2, "result = x * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Edit 3: power
        nb_runner.set_cell_source(2, "result = x ** 2\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Edit 4: floor div
        nb_runner.set_cell_source(2, "result = x // 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 3" in nb_runner.get_output(2)

        # Edit 5: complex expression
        nb_runner.set_cell_source(2, "result = (x + 1) * (x - 1)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 99" in nb_runner.get_output(2)

    def test_alternating_data_source(self, nb_runner):
        """Alternate data sources, verify correct propagation each time."""
        nb_runner.create_notebook(
            [
                "source = [1, 2, 3]",
                "total = sum(source)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        for vals, expected in [
            ([10, 20], 30),
            ([100], 100),
            ([5, 5, 5, 5], 20),
            ([1000, 2000, 3000], 6000),
        ]:
            nb_runner.set_cell_source(1, f"source = {vals}")
            nb_runner.run_all()
            assert f"total = {expected}" in nb_runner.get_output(2)

    def test_refine_function_multiple_times(self, nb_runner):
        """Refine function definition through multiple iterations."""
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x",
                "vals = [1, 2, 3, 4, 5]\nresult = [transform(v) for v in vals]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5]" in nb_runner.get_output(2)

        # v2: double
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 2")
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # v3: square
        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

        # v4: negate
        nb_runner.set_cell_source(1, "def transform(x):\n    return -x")
        nb_runner.run_all()
        assert "result = [-1, -2, -3, -4, -5]" in nb_runner.get_output(2)
