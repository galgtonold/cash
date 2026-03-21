"""Batch 387: dict inversion and bidirectional mapping."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictInversion:
    def test_invert_dict(self, nb_runner):
        nb_runner.create_notebook([
            "original = {'a': 1, 'b': 2, 'c': 3}",
            "inverted = {v: k for k, v in original.items()}\nprint(f'inverted={dict(sorted(inverted.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "inverted={1: 'a', 2: 'b', 3: 'c'}" in nb_runner.get_output(2)

    def test_invert_edit(self, nb_runner):
        nb_runner.create_notebook([
            "mapping = {'red': '#FF0000', 'green': '#00FF00'}",
            "reverse = {v: k for k, v in mapping.items()}\ncolor = reverse.get('#FF0000', 'unknown')\nprint(f'color={color}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color=red" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "mapping = {'blue': '#0000FF', 'red': '#FF0000'}")
        nb_runner.run_all()
        assert "color=red" in nb_runner.get_output(2)

    def test_groupby_inversion(self, nb_runner):
        nb_runner.create_notebook([
            "scores = {'Alice': 'A', 'Bob': 'B', 'Charlie': 'A', 'Diana': 'B'}",
            "grouped = {}\nfor name, grade in scores.items():\n    grouped.setdefault(grade, []).append(name)\nprint(f'grouped={dict(sorted(grouped.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'A': ['Alice', 'Charlie']" in out
        assert "'B': ['Bob', 'Diana']" in out
