"""
Batch 35: Multi-output cell patterns, display vs return, print ordering,
and assignment expression (walrus) patterns.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestMultiOutputCells:
    """Test cells that produce multiple outputs."""

    def test_multiple_print_statements(self, nb_runner):
        """Multiple print() in one cell."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            textwrap.dedent("""\
                print(f"x={x}")
                print(f"y={y}")
                print(f"sum={x+y}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "x=10" in output
        assert "y=20" in output
        assert "sum=30" in output

    def test_multiple_variables_assigned(self, nb_runner):
        """Multiple variables assigned in one cell, used in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                a = 1
                b = 2
                c = 3
            """),
            "total = a + b + c",
            "print(total)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(3)

    def test_mixed_computation_and_print(self, nb_runner):
        """Mix of computation and print in same cell."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                total = sum(data)
                avg = total / len(data)
                print(f"total={total} avg={avg}")
                maximum = max(data)
            """),
            textwrap.dedent("""\
                print(f"max={maximum}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15 avg=3.0" in nb_runner.get_output(2)
        assert "max=5" in nb_runner.get_output(3)


class TestWalrusOperator:
    """Test assignment expressions (:=) across cells."""

    def test_walrus_in_while_loop(self, nb_runner):
        """Walrus in while loop, used downstream."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = [1, 2, 3, 0, 4, 5]
                results = []
                idx = 0
                while (val := data[idx]) != 0:
                    results.append(val * 10)
                    idx += 1
            """),
            "print(results)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[10, 20, 30]" in nb_runner.get_output(2)

    def test_walrus_in_if(self, nb_runner):
        """Walrus in if condition."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import re
                text = "Order #12345 placed"
                if (match := re.search(r'#(\\d+)', text)):
                    order_id = match.group(1)
                else:
                    order_id = "unknown"
            """),
            "print(f'order={order_id}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "order=12345" in nb_runner.get_output(2)


class TestChainedOperations:
    """Test chained method calls and operations."""

    def test_method_chaining(self, nb_runner):
        """Method chaining across cells."""
        nb_runner.create_notebook([
            "text = '  Hello, World!  '",
            textwrap.dedent("""\
                result = text.strip().lower().replace(',', '').split()
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['hello', 'world!']" in nb_runner.get_output(2)

    def test_chained_dict_operations(self, nb_runner):
        """Chained dict operations."""
        nb_runner.create_notebook([
            "base = {'a': 1, 'b': 2}",
            "extra = {'c': 3, 'd': 4}",
            textwrap.dedent("""\
                merged = {**base, **extra}
                filtered = {k: v for k, v in merged.items() if v > 2}
                print(filtered)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'c': 3" in output
        assert "'d': 4" in output


class TestExpressionVsStatement:
    """Test expression vs statement behavior."""

    def test_bare_expression(self, nb_runner):
        """Bare expression in cell (like Jupyter shows last expression)."""
        nb_runner.create_notebook([
            "x = 42",
            "x",
            "y = x + 8",
            "print(y)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "50" in nb_runner.get_output(4)

    def test_semicolon_suppression(self, nb_runner):
        """Semicolons in cells."""
        nb_runner.create_notebook([
            "a = 1; b = 2; c = 3",
            textwrap.dedent("""\
                total = a + b + c
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)


class TestLargeVariableCount:
    """Test cells with many variables."""

    def test_many_variables_one_cell(self, nb_runner):
        """Cell producing many variables used downstream."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                v1 = 1; v2 = 2; v3 = 3; v4 = 4; v5 = 5
                v6 = 6; v7 = 7; v8 = 8; v9 = 9; v10 = 10
            """),
            textwrap.dedent("""\
                total = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "55" in nb_runner.get_output(2)

    def test_many_intermediate_variables(self, nb_runner):
        """Pipeline with many intermediate variables across cells."""
        nb_runner.create_notebook([
            "raw = list(range(10))",
            "step1 = [x + 1 for x in raw]",
            "step2 = [x * 2 for x in step1]",
            "step3 = [x for x in step2 if x > 10]",
            "step4 = sorted(step3, reverse=True)",
            textwrap.dedent("""\
                print(f"len={len(step4)} first={step4[0]} last={step4[-1]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # raw: 0-9, step1: 1-10, step2: 2,4,...,20, step3: 12,14,16,18,20, step4: 20,18,16,14,12
        output = nb_runner.get_output(6)
        assert "len=5" in output
        assert "first=20" in output
        assert "last=12" in output
