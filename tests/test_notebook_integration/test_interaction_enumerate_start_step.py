"""
Interaction test: enumerate with start parameter and custom step.
Tests enumerate with start offset, zip+enumerate patterns,
and cross-cell indexed iteration pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestEnumerateStartStep:
    """Test enumerate with start parameter across cells."""

    def test_enumerate_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: enumerate with start
            "items = ['apple', 'banana', 'cherry']\nindexed = list(enumerate(items, start=1))\nprint(f'indexed={indexed}')",
            # Cell 2: use indexed in computation
            "formatted = [f'{i}. {name}' for i, name in indexed]\nprint(f'list={formatted}')",
            # Cell 3: reversed enumerate
            "rev = list(enumerate(reversed(items), start=1))\nprint(f'reversed={rev}')\ntotal_idx = sum(i for i, _ in rev)\nprint(f'idx_sum={total_idx}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "indexed=[(1, 'apple'), (2, 'banana'), (3, 'cherry')]" in out1
        out2 = nb_runner.get_output(2)
        assert "1. apple" in out2
        assert "3. cherry" in out2
        out3 = nb_runner.get_output(3)
        assert "idx_sum=6" in out3

    def test_enumerate_edit(self, nb_runner):
        nb_runner.create_notebook([
            "colors = ['red', 'green', 'blue']\nnumbered = {i: c for i, c in enumerate(colors, 100)}\nprint(f'numbered={numbered}')",
            "keys_sum = sum(numbered.keys())\nprint(f'keys_sum={keys_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys_sum=303" in nb_runner.get_output(2)

        # Edit start value
        nb_runner.set_cell_source(1, "colors = ['red', 'green', 'blue', 'yellow']\nnumbered = {i: c for i, c in enumerate(colors, 200)}\nprint(f'numbered={numbered}')")
        nb_runner.run_cells([1, 2])
        # 200+201+202+203 = 806
        assert "keys_sum=806" in nb_runner.get_output(2)

    def test_enumerate_cache(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['hello', 'world']\nresult = [(i, w.upper()) for i, w in enumerate(words)]\nprint(f'result={result}')",
            "count = len(result)\nprint(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(0, 'HELLO'), (1, 'WORLD')]" in nb_runner.get_output(1)
        assert "count=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
