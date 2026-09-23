"""if/else and conditional expressions whose branch changes after an edit."""

import pytest

pytestmark = [pytest.mark.stress]


# Control structure + cell edit interaction tests.
#
# Tests that exercise loops and conditionals combined with cell edits,
# out-of-order execution, and kernel restarts.
@pytest.mark.control
@pytest.mark.timeout(30)
class TestLoopCellEdits:
    """Loops with cell edits."""

    def test_edit_loop_body(self, nb_runner):
        """Edit the body of a for-loop cell."""
        nb_runner.create_notebook(
            [
                "total = 0",
                "for i in range(5):\n    total += i",
                "print(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 10" in nb_runner.get_output(3)

        # Edit loop body: multiply instead of add
        nb_runner.set_cell_source(2, "for i in range(5):\n    total += i * 2")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 20" in nb_runner.get_output(3)

    def test_edit_loop_range(self, nb_runner):
        """Change the range of a loop."""
        nb_runner.create_notebook(
            [
                "total = 0",
                "for i in range(3):\n    total += i",
                "print(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "for i in range(10):\n    total += i")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(3)

    def test_edit_loop_init_value(self, nb_runner):
        """Change the initial value before a loop."""
        nb_runner.create_notebook(
            [
                "total = 100",
                "for i in range(3):\n    total += i",
                "print(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 103" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "total = 0")
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(3)

    def test_nested_loop_edit(self, nb_runner):
        """Edit a nested loop."""
        nb_runner.create_notebook(
            [
                "pairs = []\nfor i in range(3):\n    for j in range(2):\n        pairs.append((i, j))",
                "print(f'count = {len(pairs)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "pairs = []\nfor i in range(4):\n    for j in range(3):\n        pairs.append((i, j))",
        )
        nb_runner.run_all()
        assert "count = 12" in nb_runner.get_output(2)


@pytest.mark.control
@pytest.mark.timeout(30)
class TestConditionalCellEdits:
    """Conditionals with cell edits."""

    def test_edit_branch_bodies(self, nb_runner):
        """Edit what the branches produce."""
        nb_runner.create_notebook(
            [
                "flag = True",
                "if flag:\n    val = 'yes'\nelse:\n    val = 'no'",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = yes" in nb_runner.get_output(3)

        # Edit branch bodies
        nb_runner.set_cell_source(2, "if flag:\n    val = 'TRUE'\nelse:\n    val = 'FALSE'")
        nb_runner.run_all()
        assert "val = TRUE" in nb_runner.get_output(3)

    def test_flip_flag_multiple_times(self, nb_runner):
        """Flip a boolean flag back and forth."""
        nb_runner.create_notebook(
            [
                "flag = True",
                "result = 'A' if flag else 'B'\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = A" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "flag = False")
        nb_runner.run_all()
        assert "result = B" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "flag = True")
        nb_runner.run_all()
        assert "result = A" in nb_runner.get_output(2)

    def test_conditional_with_function_call(self, nb_runner):
        """Conditional that calls a function, both get edited."""
        nb_runner.create_notebook(
            [
                "def classify(n):\n    return 'even' if n % 2 == 0 else 'odd'",
                "x = 4",
                "label = classify(x)\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = even" in nb_runner.get_output(3)

        # Edit both: change function and value
        nb_runner.set_cell_source(1, "def classify(n):\n    return 'pos' if n > 0 else 'neg'")
        nb_runner.set_cell_source(2, "x = -3")
        nb_runner.run_all()
        assert "label = neg" in nb_runner.get_output(3)


@pytest.mark.control
@pytest.mark.timeout(30)
class TestLoopRerunConsistency:
    """Loops must not accumulate on rerun."""

    def test_for_loop_no_double_count(self, nb_runner):
        """Re-running a loop cell should give same result."""
        nb_runner.create_notebook(
            [
                "items = []",
                "for i in range(3):\n    items.append(i)",
                "print(f'items = {items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [0, 1, 2]" in nb_runner.get_output(3)

        # Re-run
        nb_runner.run_all()
        assert "items = [0, 1, 2]" in nb_runner.get_output(3)

    def test_while_loop_edit_condition(self, nb_runner):
        """Edit while loop condition."""
        nb_runner.create_notebook(
            [
                "i = 0\nresult = []",
                "while i < 3:\n    result.append(i)\n    i += 1",
                "print(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2]" in nb_runner.get_output(3)

        # Change condition to < 5
        nb_runner.set_cell_source(2, "while i < 5:\n    result.append(i)\n    i += 1")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3, 4]" in nb_runner.get_output(3)


# Multi-cell conditional branching with shared state.
#
# Tests where conditional logic spans multiple cells, with shared
# state that changes based on which branch was taken, and edits
# to the condition or branch bodies.
@pytest.mark.control
@pytest.mark.timeout(90)
class TestConditionalStateEdits:
    """Conditional branching affecting shared state."""

    def test_edit_condition_value(self, nb_runner):
        """Edit the condition variable, verify different branch taken."""
        nb_runner.create_notebook(
            [
                "mode = 'fast'  # processing mode",
                "if mode == 'fast':\n    factor = 10\nelse:\n    factor = 1",
                "result = 42 * factor\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 420" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "mode = 'slow'  # processing mode changed")
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(3)

    def test_edit_branch_body(self, nb_runner):
        """Edit a branch body."""
        nb_runner.create_notebook(
            [
                "flag = True  # branch flag",
                "if flag:\n    val = 'YES'\nelse:\n    val = 'NO'",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = YES" in nb_runner.get_output(3)

        # Edit the true branch
        nb_runner.set_cell_source(2, "if flag:\n    val = 'AFFIRMATIVE'\nelse:\n    val = 'NEGATIVE'")
        nb_runner.run_all()
        assert "val = AFFIRMATIVE" in nb_runner.get_output(3)

    def test_add_elif_branch(self, nb_runner):
        """Add an elif branch to existing if/else."""
        nb_runner.create_notebook(
            [
                "level = 5  # level value",
                "if level > 10:\n    label = 'high'\nelse:\n    label = 'low'",
                "print(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = low" in nb_runner.get_output(3)

        # Add elif
        nb_runner.set_cell_source(
            2,
            "if level > 10:\n    label = 'high'\nelif level > 3:\n    label = 'medium'\nelse:\n    label = 'low'",
        )
        nb_runner.run_all()
        assert "label = medium" in nb_runner.get_output(3)


@pytest.mark.control
@pytest.mark.timeout(90)
class TestMultiCellBranching:
    """Branching logic split across multiple cells."""

    def test_config_driven_pipeline(self, nb_runner):
        """Config cell drives processing in multiple downstream cells."""
        nb_runner.create_notebook(
            [
                "config = {'scale': 2, 'offset': 10}  # config dict",
                "scaled = 5 * config['scale']",
                "result = scaled + config['offset']\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

        # Edit config
        nb_runner.set_cell_source(1, "config = {'scale': 5, 'offset': 100}  # config dict updated")
        nb_runner.run_all()
        assert "result = 125" in nb_runner.get_output(3)

    def test_flag_toggle_multiple_cells(self, nb_runner):
        """Toggle a flag that affects multiple downstream cells."""
        nb_runner.create_notebook(
            [
                "use_prefix = True  # flag for prefix",
                "if use_prefix:\n    label = 'ENABLED'\nelse:\n    label = 'DISABLED'\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = ENABLED" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "use_prefix = False  # flag for prefix off")
        nb_runner.run_all()
        assert "label = DISABLED" in nb_runner.get_output(2)

    def test_switch_case_pattern(self, nb_runner):
        """Dictionary-based switch/case pattern with edits."""
        nb_runner.create_notebook(
            [
                "action = 'add'  # action selector",
                "ops = {'add': lambda a, b: a + b, 'mul': lambda a, b: a * b}",
                "result = ops[action](3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "action = 'mul'  # action selector changed")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)


# Conditional logic and boolean pattern interaction tests.
#
# Tests where users edit conditional logic (if/elif/else),
# boolean variables, and branch selection patterns.
@pytest.mark.control
@pytest.mark.timeout(45)
class TestIfElseEdits:
    """If/else editing patterns."""

    def test_edit_branch_logic(self, nb_runner):
        """Edit the branch logic itself."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "result = 'positive' if x > 0 else 'non-positive'\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = positive" in nb_runner.get_output(2)

        # Change the logic
        nb_runner.set_cell_source(
            2,
            "result = 'even' if x % 2 == 0 else 'odd'\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = even" in nb_runner.get_output(2)


@pytest.mark.control
@pytest.mark.timeout(45)
class TestBooleanEdits:
    """Boolean flag editing patterns."""

    def test_toggle_boolean_flag(self, nb_runner):
        """Toggle a boolean flag."""
        nb_runner.create_notebook(
            [
                "debug = True",
                "data = [1, 2, 3]\nif debug:\n    data = [x * 10 for x in data]",
                "total = sum(data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(3)

        # Toggle debug off
        nb_runner.set_cell_source(1, "debug = False")
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(3)

    def test_multiple_boolean_flags(self, nb_runner):
        """Multiple boolean flags controlling logic."""
        nb_runner.create_notebook(
            [
                "normalize = True\nround_result = True",
                "raw = [10, 20, 30]\nif normalize:\n    vals = [x / max(raw) for x in raw]\nelse:\n    vals = raw",
                "if round_result:\n    vals = [round(v, 1) for v in vals]",
                "print(f'vals = {vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "vals = " in output

        # Turn off normalize
        nb_runner.set_cell_source(1, "normalize = False\nround_result = True")
        nb_runner.run_all()
        assert "vals = [10, 20, 30]" in nb_runner.get_output(4)


@pytest.mark.control
@pytest.mark.timeout(45)
class TestNestedConditionalEdits:
    """Nested conditional patterns with edits."""

    def test_edit_outer_condition(self, nb_runner):
        """Edit outer condition in nested if."""
        nb_runner.create_notebook(
            [
                "mode = 'fast'\nverbose = True",
                "if mode == 'fast':\n    result = 100\nelif mode == 'slow':\n    result = 1\nelse:\n    result = 10",
                "if verbose:\n    msg = f'Mode: {mode}, Result: {result}'\nelse:\n    msg = str(result)\nprint(msg)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Mode: fast, Result: 100" in nb_runner.get_output(3)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'slow'\nverbose = True")
        nb_runner.run_all()
        assert "Mode: slow, Result: 1" in nb_runner.get_output(3)

    def test_edit_inner_condition(self, nb_runner):
        """Edit inner condition flag."""
        nb_runner.create_notebook(
            [
                "mode = 'fast'\nverbose = True",
                "if mode == 'fast':\n    result = 100\nelse:\n    result = 1",
                "if verbose:\n    msg = f'Result: {result}'\nelse:\n    msg = 'done'\nprint(msg)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Result: 100" in nb_runner.get_output(3)

        # Turn off verbose
        nb_runner.set_cell_source(1, "mode = 'fast'\nverbose = False")
        nb_runner.run_all()
        assert "done" in nb_runner.get_output(3)


# Conditional logic and branching edit tests.
#
# Tests editing cells with conditional logic to switch between
# branches and verify cache handles the change correctly.
@pytest.mark.control
@pytest.mark.timeout(90)
class TestConditionalBranchEdits:
    """Editing conditional branching patterns."""

    def test_edit_if_condition_flip(self, nb_runner):
        """Flip an if condition from True to False."""
        nb_runner.create_notebook(
            [
                "threshold = 50\nvalue = 75",
                "if value > threshold:\n    label = 'above'\nelse:\n    label = 'below'\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = above" in nb_runner.get_output(2)

        # Change value to be below threshold
        nb_runner.set_cell_source(1, "threshold = 50\nvalue = 25")
        nb_runner.run_all()
        assert "label = below" in nb_runner.get_output(2)

    def test_edit_elif_chain(self, nb_runner):
        """Edit value to hit a different elif branch."""
        nb_runner.create_notebook(
            [
                "score = 85",
                "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelif score >= 70:\n    grade = 'C'\nelse:\n    grade = 'F'\nprint(f'grade = {grade}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade = B" in nb_runner.get_output(2)

        # Change to grade A
        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "grade = A" in nb_runner.get_output(2)

    def test_edit_ternary_expression(self, nb_runner):
        """Edit a ternary expression's condition."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "label = 'positive' if x > 0 else 'non-positive'\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = positive" in nb_runner.get_output(2)

        # Change to negative
        nb_runner.set_cell_source(1, "x = -5")
        nb_runner.run_all()
        assert "label = non-positive" in nb_runner.get_output(2)

    def test_edit_nested_condition(self, nb_runner):
        """Edit a nested if/else pattern."""
        nb_runner.create_notebook(
            [
                "age = 25\nhas_license = True",
                "if age >= 18:\n    if has_license:\n        status = 'can drive'\n    else:\n        status = 'needs license'\nelse:\n    status = 'too young'\nprint(f'status = {status}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "status = can drive" in nb_runner.get_output(2)

        # Remove license
        nb_runner.set_cell_source(1, "age = 25\nhas_license = False")
        nb_runner.run_all()
        assert "status = needs license" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestConditionalExprChain:
    """conditional expression chains and ternary nesting."""

    def test_ternary_chain(self, nb_runner):
        nb_runner.create_notebook(
            [
                "score = 85",
                "grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'F'\nprint(f'grade={grade}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade=B" in nb_runner.get_output(2)

    def test_ternary_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "val = 15",
                "category = 'high' if val > 10 else 'low'\nprint(f'category={category}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "category=high" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "val = 5")
        nb_runner.run_all()
        assert "category=low" in nb_runner.get_output(2)

    def test_conditional_comprehension(self, nb_runner):
        nb_runner.create_notebook(
            [
                "values = list(range(-5, 6))",
                "labels = ['pos' if x > 0 else 'neg' if x < 0 else 'zero' for x in values]\nzero_count = labels.count('zero')\npos_count = labels.count('pos')\nprint(f'zeros={zero_count} pos={pos_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "zeros=1 pos=5" in nb_runner.get_output(2)


# Ternary and inline conditional edit propagation.
#
# Tests ternary expressions and inline conditionals.
@pytest.mark.timeout(90)
class TestTernaryEdits:
    """Ternary/inline conditional patterns."""

    def test_ternary_condition_edit(self, nb_runner):
        """Edit condition in ternary expression."""
        nb_runner.create_notebook(
            [
                "threshold = 50",
                "value = 75",
                "label = 'high' if value > threshold else 'low'\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = high" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "threshold = 80")
        nb_runner.run_all()
        assert "label = low" in nb_runner.get_output(3)

    def test_chained_ternary_edit(self, nb_runner):
        """Edit value with chained ternary."""
        nb_runner.create_notebook(
            [
                "score = 85",
                "grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'F'\nprint(f'grade = {grade}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade = B" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "grade = A" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "score = 65")
        nb_runner.run_all()
        assert "grade = F" in nb_runner.get_output(2)

    def test_list_comp_with_conditional_edit(self, nb_runner):
        """Edit filter condition in list comprehension."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "limit = 5",
                "filtered = [x for x in data if x > limit]\nprint(f'filtered = {filtered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filtered = [6, 7, 8, 9, 10]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "limit = 8")
        nb_runner.run_all()
        assert "filtered = [9, 10]" in nb_runner.get_output(3)
