"""Typing module & Protocol patterns — cash caching with type annotations."""

import textwrap

import pytest


@pytest.mark.stress
class TestProtocolPatterns:
    """Test Protocol-based structural typing."""

    def test_typed_change_propagation(self, nb_runner):
        """Type-annotated variables propagate on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from typing import Dict, List

                scores: Dict[str, List[int]] = {
                    'math': [90, 85, 92],
                    'science': [88, 91, 87],
                }
            """),
                textwrap.dedent("""\
                averages = {k: sum(v) / len(v) for k, v in scores.items()}
                print(f"averages={averages}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "math" in out
        assert "89.0" in out

        # Add a subject
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from typing import Dict, List

            scores: Dict[str, List[int]] = {
                'math': [90, 85, 92],
                'science': [88, 91, 87],
                'english': [95, 90, 88],
            }
        """),
        )
        nb_runner.run_cells([1, 2])
        out2 = nb_runner.get_output(2)
        assert "english" in out2
        assert "91.0" in out2
