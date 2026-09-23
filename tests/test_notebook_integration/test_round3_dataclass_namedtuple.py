"""Dataclass & NamedTuple advanced patterns — cash caching with typed data."""

import textwrap

import pytest


@pytest.mark.stress
class TestDataclassAdvanced:
    """Test advanced dataclass patterns."""

    def test_dataclass_change_propagates(self, nb_runner):
        """Changing dataclass definition propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from dataclasses import dataclass

                @dataclass
                class Config:
                    name: str
                    value: int = 0
            """),
                textwrap.dedent("""\
                c = Config("test", 10)
                print(f"c={c}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name='test'" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from dataclasses import dataclass

            @dataclass
            class Config:
                name: str
                value: int = 0
                active: bool = True
        """),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name='test'" in out
        assert "active=True" in out
