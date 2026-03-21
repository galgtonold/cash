"""Batch 437: walrus operator and conditional expressions."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestWalrusConditionalExpr:
    def test_walrus_basic(self, nb_runner):
        nb_runner.create_notebook([
            "data = [1, 5, 3, 8, 2, 7]",
            "big = [y for x in data if (y := x * 2) > 8]\nprint(f'big={big}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big=[10, 16, 14]" in nb_runner.get_output(2)

    def test_conditional_expr(self, nb_runner):
        nb_runner.create_notebook([
            "x = 15",
            "label = 'even' if x % 2 == 0 else 'odd'\nsign = 'positive' if x > 0 else 'non-positive'\nprint(f'label={label} sign={sign}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label=odd" in nb_runner.get_output(2)
        assert "sign=positive" in nb_runner.get_output(2)

    def test_walrus_edit(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'hello world'",
            "if (n := len(text)) > 5:\n    result = f'long ({n})'\nelse:\n    result = f'short ({n})'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=long (11)" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "text = 'hi'")
        nb_runner.run_all()
        assert "result=short (2)" in nb_runner.get_output(2)
