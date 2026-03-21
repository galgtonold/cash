"""
Interaction test: dict comprehension with conditional expressions.
Tests dict comprehension with ternary operators, nested conditions,
and cross-cell dict transformation pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictCompConditionalExpr:
    """Test dict comprehension with conditional expressions across cells."""

    def test_dict_comp_ternary(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: dict comp with ternary
            "scores = {'Alice': 85, 'Bob': 62, 'Charlie': 91, 'Diana': 45, 'Eve': 78}\ngrades = {name: ('pass' if score >= 60 else 'fail') for name, score in scores.items()}\nprint(f'grades={grades}')",
            # Cell 2: filter and transform
            "passing = {k: v for k, v in scores.items() if grades[k] == 'pass'}\navg_pass = sum(passing.values()) / len(passing)\nprint(f'passing_count={len(passing)}')\nprint(f'avg_pass={avg_pass:.1f}')",
            # Cell 3: categorize
            "categories = {name: ('A' if s >= 90 else 'B' if s >= 80 else 'C' if s >= 70 else 'D' if s >= 60 else 'F') for name, s in scores.items()}\nprint(f'categories={categories}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'Alice': 'pass'" in out1
        assert "'Diana': 'fail'" in out1
        out2 = nb_runner.get_output(2)
        assert "passing_count=4" in out2
        out3 = nb_runner.get_output(3)
        assert "'Charlie': 'A'" in out3
        assert "'Alice': 'B'" in out3
        assert "'Diana': 'F'" in out3

    def test_dict_comp_edit(self, nb_runner):
        nb_runner.create_notebook([
            "prices = {'apple': 1.5, 'banana': 0.5, 'cherry': 3.0}\ndiscounted = {k: round(v * 0.9, 2) for k, v in prices.items()}\nprint(f'disc={discounted}')",
            "total = sum(discounted.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Change discount rate
        nb_runner.set_cell_source(1, "prices = {'apple': 1.5, 'banana': 0.5, 'cherry': 3.0}\ndiscounted = {k: round(v * 0.8, 2) for k, v in prices.items()}\nprint(f'disc={discounted}')")
        nb_runner.run_cells([1, 2])
        assert "'apple': 1.2" in nb_runner.get_output(1)
        assert "total=4.0" in nb_runner.get_output(2)

    def test_dict_comp_cache(self, nb_runner):
        nb_runner.create_notebook([
            "data = [('a', 1), ('b', 2), ('c', 3)]\nd = {k: v ** 2 for k, v in data}\nprint(f'd={d}')",
            "total = sum(d.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d={'a': 1, 'b': 4, 'c': 9}" in nb_runner.get_output(1)
        assert "total=14" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=14" in nb_runner.get_output(2)
