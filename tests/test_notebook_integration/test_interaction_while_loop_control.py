"""Batch 357: while loop with break/continue and state accumulation."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestWhileLoopControl:
    def test_while_break(self, nb_runner):
        nb_runner.create_notebook([
            "threshold = 100",
            "total = 0\ni = 1\nwhile True:\n    total += i\n    if total >= threshold:\n        break\n    i += 1\nprint(f'total={total} i={i}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=" in nb_runner.get_output(2)
        assert "i=" in nb_runner.get_output(2)

    def test_while_continue_edit(self, nb_runner):
        nb_runner.create_notebook([
            "limit = 10",
            "evens = []\ni = 0\nwhile i < limit:\n    i += 1\n    if i % 2 != 0:\n        continue\n    evens.append(i)\nprint(f'evens={evens}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens=[2, 4, 6, 8, 10]" in nb_runner.get_output(2)
        # Edit limit
        nb_runner.set_cell_source(1, "limit = 6")
        nb_runner.run_all()
        assert "evens=[2, 4, 6]" in nb_runner.get_output(2)

    def test_while_accumulate(self, nb_runner):
        nb_runner.create_notebook([
            "n = 5",
            "factorial = 1\ncurrent = n\nwhile current > 1:\n    factorial *= current\n    current -= 1\nprint(f'factorial={factorial}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "factorial=120" in nb_runner.get_output(2)
