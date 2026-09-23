"""Itertools advanced — cash caching with itertools combinatorial patterns."""

import textwrap

import pytest


@pytest.mark.stress
class TestItertoolsInfinite:
    """Test infinite iterator patterns."""

    def test_itertools_propagation(self, nb_runner):
        """Itertools results propagate on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from itertools import combinations
                items = ['A', 'B', 'C']
                pairs = list(combinations(items, 2))
            """),
                textwrap.dedent("""\
                labels = [f"{a}-{b}" for a, b in pairs]
                print(f"labels={labels}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "labels=['A-B', 'A-C', 'B-C']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from itertools import combinations
            items = ['X', 'Y', 'Z', 'W']
            pairs = list(combinations(items, 2))
        """),
        )
        nb_runner.run_cells([1, 2])
        out = nb_runner.get_output(2)
        assert "X-Y" in out
        assert "Z-W" in out
