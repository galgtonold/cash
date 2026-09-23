"""Statistics & random distributions — cash caching with statistical computations."""

import textwrap

import pytest


@pytest.mark.stress
class TestStatisticsModule:
    """Test statistics module patterns across cells."""

    def test_statistics_propagation(self, nb_runner):
        """Statistical results propagate on data change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = [10, 20, 30, 40, 50]
            """),
                textwrap.dedent("""\
                import statistics
                mean_val = statistics.mean(data)
                print(f"mean={mean_val}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            data = [100, 200, 300, 400, 500]
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "mean=300" in nb_runner.get_output(2)
