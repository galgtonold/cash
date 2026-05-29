"""
Stress Test Batch 6: If/Else Branch Regression Tests (166-180)

These specifically test the if/else branch switch bug that was found and fixed
in _simulate_if in upstream.py. The bug was that the upstream simulation evaluated
if/else conditions using stale user_ns values when the condition variable was
modified in an upstream cell.

The fix: when a condition variable's virtual lineage differs from its actual
lineage, fall back to treating the entire if/else as a monolithic statement.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.control]


class TestIfElseBranchRegression:
    """Regression tests for the if/else branch switch bug."""

    def test_166_elif_chain_branch_switch(self, nb_runner):
        """
        Cell 1: mode = 'a'
        Cell 2: if mode == 'a': x = 1; elif mode == 'b': x = 2; else: x = 3
        Cell 3: print(x)
        
        Switch mode to 'b', run only cell 3. Should get x=2.
        """
        nb_runner.create_notebook([
            "mode = 'a'",
            "if mode == 'a':\n    x = 1\nelif mode == 'b':\n    x = 2\nelse:\n    x = 3",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=1" in nb_runner.get_output(3)
        # Switch to 'b'
        nb_runner.set_cell_source(1, "mode = 'b'")
        nb_runner.run_cell(3)
        assert "x=2" in nb_runner.get_output(3)


    def test_168_numeric_condition_change(self, nb_runner):
        """Condition based on numeric comparison."""
        nb_runner.create_notebook([
            "threshold = 50",
            "score = 75",
            "if score >= threshold:\n    grade = 'pass'\nelse:\n    grade = 'fail'",
            "print(f'grade={grade}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade=pass" in nb_runner.get_output(4)
        # Raise threshold so score fails
        nb_runner.set_cell_source(1, "threshold = 80")
        nb_runner.run_cell(4)
        assert "grade=fail" in nb_runner.get_output(4)

    def test_169_boolean_flag_toggle_back_and_forth(self, nb_runner):
        """Toggle flag True→False→True, verify each switch works."""
        nb_runner.create_notebook([
            "is_verbose = True",
            "if is_verbose:\n    msg = 'VERBOSE ON'\nelse:\n    msg = 'VERBOSE OFF'",
            "print(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=VERBOSE ON" in nb_runner.get_output(3)
        # Toggle to False
        nb_runner.set_cell_source(1, "is_verbose = False")
        nb_runner.run_cell(3)
        assert "msg=VERBOSE OFF" in nb_runner.get_output(3)
        # Toggle back to True
        nb_runner.set_cell_source(1, "is_verbose = True")
        nb_runner.run_cell(3)
        assert "msg=VERBOSE ON" in nb_runner.get_output(3)

    def test_170_if_else_multiple_outputs(self, nb_runner):
        """If/else that sets multiple variables in each branch."""
        nb_runner.create_notebook([
            "use_metric = True",
            "if use_metric:\n    unit = 'km'\n    factor = 1.0\nelse:\n    unit = 'miles'\n    factor = 0.621371",
            "result = 100 * factor\nprint(f'{result} {unit}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100.0 km" in nb_runner.get_output(3)
        # Switch to imperial
        nb_runner.set_cell_source(1, "use_metric = False")
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "miles" in out
        assert "62.1371" in out

    def test_171_nested_if_with_changed_outer_condition(self, nb_runner):
        """Nested if where outer condition changes."""
        nb_runner.create_notebook([
            "level = 'high'",
            "if level == 'high':\n    if True:\n        x = 'HIGH'\nelse:\n    x = 'LOW'",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=HIGH" in nb_runner.get_output(3)
        # Change to low
        nb_runner.set_cell_source(1, "level = 'low'")
        nb_runner.run_cell(3)
        assert "x=LOW" in nb_runner.get_output(3)

    def test_172_if_condition_with_expression(self, nb_runner):
        """Condition uses an expression, not just a variable."""
        nb_runner.create_notebook([
            "n = 10",
            "if n > 5:\n    category = 'big'\nelse:\n    category = 'small'",
            "print(f'category={category}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "category=big" in nb_runner.get_output(3)
        # Change n to be small
        nb_runner.set_cell_source(1, "n = 3")
        nb_runner.run_cell(3)
        assert "category=small" in nb_runner.get_output(3)

    def test_173_if_condition_with_two_variables(self, nb_runner):
        """Condition uses two variables: if a > b."""
        nb_runner.create_notebook([
            "a = 10\nb = 5",
            "if a > b:\n    result = 'a wins'\nelse:\n    result = 'b wins'",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=a wins" in nb_runner.get_output(3)
        # Swap — make b bigger
        nb_runner.set_cell_source(1, "a = 3\nb = 5")
        nb_runner.run_cell(3)
        assert "result=b wins" in nb_runner.get_output(3)

    def test_174_if_only_no_else_branch(self, nb_runner):
        """
        If without else — variable only set in if-branch.
        When condition becomes False, variable may not be set by if at all.
        """
        nb_runner.create_notebook([
            "flag = True",
            "x = 'default'\nif flag:\n    x = 'overridden'",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=overridden" in nb_runner.get_output(3)
        # Change flag to False — x should stay 'default'
        nb_runner.set_cell_source(1, "flag = False")
        nb_runner.run_cell(3)
        assert "x=default" in nb_runner.get_output(3)

    def test_175_if_else_in_loop_condition_changes(self, nb_runner):
        """If/else inside a loop, where the condition variable changes upstream."""
        nb_runner.create_notebook([
            "multiplier = 2",
            "results = []\nfor i in range(3):\n    if multiplier > 1:\n        results.append(i * multiplier)\n    else:\n        results.append(i)",
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 2, 4]" in nb_runner.get_output(3)
        # Change multiplier
        nb_runner.set_cell_source(1, "multiplier = 0")
        nb_runner.run_all()
        assert "results=[0, 1, 2]" in nb_runner.get_output(3)

    def test_176_if_else_with_function_call_in_condition(self, nb_runner):
        """Condition calls a function defined in upstream cell."""
        nb_runner.create_notebook([
            "def check():\n    return True",
            "if check():\n    status = 'OK'\nelse:\n    status = 'FAIL'",
            "print(f'status={status}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "status=OK" in nb_runner.get_output(3)
        # Change function to return False
        nb_runner.set_cell_source(1, "def check():\n    return False")
        nb_runner.run_cell(3)
        assert "status=FAIL" in nb_runner.get_output(3)

    def test_177_if_condition_unchanged_body_changed(self, nb_runner):
        """
        Condition stays same, but body code changes.
        Should detect body change and re-execute.
        """
        nb_runner.create_notebook([
            "flag = True",
            "if flag:\n    x = 'original'\nelse:\n    x = 'nope'",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=original" in nb_runner.get_output(3)
        # Change the body (not the condition)
        nb_runner.set_cell_source(2, "if flag:\n    x = 'modified'\nelse:\n    x = 'nope'")
        nb_runner.run_cell(3)
        assert "x=modified" in nb_runner.get_output(3)

    def test_178_if_else_with_side_effect(self, nb_runner):
        """If/else where branch has a print side effect."""
        nb_runner.create_notebook([
            "verbose = True",
            "if verbose:\n    print('VERBOSE: executing')\n    x = 42\nelse:\n    x = 0",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=42" in nb_runner.get_output(3)
        # Disable verbose
        nb_runner.set_cell_source(1, "verbose = False")
        nb_runner.run_cell(3)
        assert "x=0" in nb_runner.get_output(3)

    def test_179_chained_if_else_cells(self, nb_runner):
        """Multiple cells with if/else, all depending on same flag."""
        nb_runner.create_notebook([
            "mode = 'fast'",
            "if mode == 'fast':\n    batch_size = 1000\nelse:\n    batch_size = 10",
            "if mode == 'fast':\n    timeout = 5\nelse:\n    timeout = 60",
            "print(f'batch={batch_size}, timeout={timeout}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "batch=1000" in nb_runner.get_output(4)
        assert "timeout=5" in nb_runner.get_output(4)
        # Switch mode
        nb_runner.set_cell_source(1, "mode = 'safe'")
        nb_runner.run_cell(4)
        out = nb_runner.get_output(4)
        assert "batch=10" in out
        assert "timeout=60" in out

    def test_180_if_else_diamond_dependency(self, nb_runner):
        """
        Cell 1: flag = True
        Cell 2: if flag: a = 1 else: a = 2
        Cell 3: b = a + 10  (depends on a, which comes from if/else)
        Cell 4: print(b)
        
        Change flag, run cell 4 only.
        """
        nb_runner.create_notebook([
            "flag = True",
            "if flag:\n    a = 1\nelse:\n    a = 2",
            "b = a + 10",
            "print(f'b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b=11" in nb_runner.get_output(4)
        # Change flag
        nb_runner.set_cell_source(1, "flag = False")
        nb_runner.run_cell(4)
        assert "b=12" in nb_runner.get_output(4)
