"""
Weakref, copy, and memory management patterns across cells.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestCopyPatterns:
    """Test shallow/deep copy across cells."""

    def test_copy_propagation_on_change(self, nb_runner):
        """Change original → copy is independent."""
        nb_runner.create_notebook(
            [
                "import copy",
                "data = [10, 20, 30]",
                "snapshot = copy.deepcopy(data)",
                textwrap.dedent("""\
                print(f"data={data} snap={snapshot}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "data=[10, 20, 30] snap=[10, 20, 30]" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "data = [100, 200, 300]")
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "data=[100, 200, 300]" in output
        assert "snap=[100, 200, 300]" in output
