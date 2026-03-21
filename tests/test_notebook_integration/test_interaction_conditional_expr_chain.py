"""Batch 373: conditional expression chains and ternary nesting."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestConditionalExprChain:
    def test_ternary_chain(self, nb_runner):
        nb_runner.create_notebook([
            "score = 85",
            "grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'F'\nprint(f'grade={grade}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade=B" in nb_runner.get_output(2)

    def test_ternary_edit(self, nb_runner):
        nb_runner.create_notebook([
            "val = 15",
            "category = 'high' if val > 10 else 'low'\nprint(f'category={category}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "category=high" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "val = 5")
        nb_runner.run_all()
        assert "category=low" in nb_runner.get_output(2)

    def test_conditional_comprehension(self, nb_runner):
        nb_runner.create_notebook([
            "values = list(range(-5, 6))",
            "labels = ['pos' if x > 0 else 'neg' if x < 0 else 'zero' for x in values]\nzero_count = labels.count('zero')\npos_count = labels.count('pos')\nprint(f'zeros={zero_count} pos={pos_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "zeros=1 pos=5" in nb_runner.get_output(2)
