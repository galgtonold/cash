"""Bug-hunt round 2: upstream-edit propagation (cash's core value prop).

Edit an upstream cell, then run a DOWNSTREAM cell only; cash must auto-re-execute
the changed upstream and propagate the new value. Devious edge cases.
"""

import pytest


class TestUpstreamEditPropagation:

    def test_edit_constant_transitive_3_levels(self, nb_runner):
        """c1: k=2 ; c2: a=k*10 ; c3: b=a+1 ; c4: print(b). Edit k=5, run c4 only."""
        nb_runner.create_notebook([
            "k = 2",
            "a = k * 10",
            "b = a + 1",
            "print(f'b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b=21" in nb_runner.get_output(4)
        nb_runner.set_cell_source(1, "k = 5")
        nb_runner.run_cell(4)  # only the leaf — cash must re-run c1,c2,c3
        assert "b=51" in nb_runner.get_output(4), f"got {nb_runner.get_output(4)!r}"

    def test_edit_function_body_downstream_recomputes(self, nb_runner):
        """Change a function's body; a downstream cell that calls it must recompute."""
        nb_runner.create_notebook([
            "def f(x):\n    return x * 2",
            "y = f(10)",
            "print(f'y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y=20" in nb_runner.get_output(3)
        nb_runner.set_cell_source(1, "def f(x):\n    return x * 3")
        nb_runner.run_cell(3)
        assert "y=30" in nb_runner.get_output(3), f"got {nb_runner.get_output(3)!r}"

    def test_shadowed_variable_edit_earlier_def(self, nb_runner):
        """Shadowing + intermediate dependency: x defined in c1 and c3; y (c2)
        depends on c1's x. Editing c1 and re-running the leaf c5 must give
        2,109,202 (matching run_all), not 2,102,202. Fixed by the planner's
        shadow-completion pass: re-materialise every version of a shadowed
        variable consumed at a stale version."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 100",
            "x = 2",
            "z = x + 200",
            "print(f'{x},{y},{z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "2,101,202" in nb_runner.get_output(5)
        nb_runner.set_cell_source(1, "x = 9")
        nb_runner.run_cell(5)
        # y depends on c1's x (=9 now -> 109); x/z depend on c3's x (=2, unchanged)
        assert "2,109,202" in nb_runner.get_output(5), f"got {nb_runner.get_output(5)!r}"

    def test_edit_value_used_in_downstream_loop(self, nb_runner):
        """Edit an upstream constant consumed inside a downstream for-loop."""
        nb_runner.create_notebook([
            "n = 3",
            "acc = 0\nfor i in range(n):\n    acc += i\nprint(f'acc={acc}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "acc=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "n = 5")
        nb_runner.run_cell(2)
        assert "acc=10" in nb_runner.get_output(2), f"got {nb_runner.get_output(2)!r}"

    def test_edit_upstream_type_change_propagates(self, nb_runner):
        """Upstream var changes type (list -> int); downstream must recompute/error correctly."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "result = len(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = [1, 2, 3, 4, 5]")
        nb_runner.run_cell(2)
        assert "result=5" in nb_runner.get_output(2), f"got {nb_runner.get_output(2)!r}"

    def test_edit_constant_inside_lambda_downstream(self, nb_runner):
        """Constant captured by a lambda defined downstream; edit constant, re-run."""
        nb_runner.create_notebook([
            "mult = 2",
            "f = lambda x: x * mult\nout = f(10)\nprint(f'out={out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "mult = 4")
        nb_runner.run_cell(2)
        assert "out=40" in nb_runner.get_output(2), f"got {nb_runner.get_output(2)!r}"
