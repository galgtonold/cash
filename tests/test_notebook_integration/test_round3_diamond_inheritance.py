"""complex inheritance: diamonds, MRO, super() chains."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDiamondInheritance:
    """Diamond inheritance patterns and MRO."""

    def test_diamond_propagation(self, nb_runner):
        """Change in base class propagates through diamond."""
        nb_runner.create_notebook(
            [
                "base_label = 'v1'",
                textwrap.dedent("""\
                class Base:
                    label = base_label
                class Left(Base): pass
                class Right(Base): pass
                class Diamond(Left, Right): pass
                d = Diamond()
            """),
                "print(f'label={d.label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label=v1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "base_label = 'v2'")
        nb_runner.run_cells([1, 2, 3])
        assert "label=v2" in nb_runner.get_output(3)
