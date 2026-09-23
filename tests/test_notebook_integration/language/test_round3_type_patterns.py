"""
Type system and typing patterns — type hints, TypeVar, Generic,
Protocol, Union, Optional, Literal across cells.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestTypeHintPatterns:
    """Test caching with type annotations in code."""

    def test_optional_annotation(self, nb_runner):
        """Optional type annotation."""
        nb_runner.create_notebook(
            [
                "from typing import Optional",
                textwrap.dedent("""\
                def find(items: list, key: str) -> Optional[int]:
                    for i, item in enumerate(items):
                        if item == key:
                            return i
                    return None
            """),
                textwrap.dedent("""\
                idx = find(['a', 'b', 'c'], 'b')
                miss = find(['a', 'b', 'c'], 'z')
                print(f"idx={idx} miss={miss}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "idx=1 miss=None" in nb_runner.get_output(3)


class TestCallableTypePatterns:
    """Test Callable type patterns."""

    def test_callable_change_propagation(self, nb_runner):
        """Change callable parameter → output updates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def transform(fn, data):
                    return [fn(x) for x in data]
            """),
                "op = lambda x: x + 1",
                textwrap.dedent("""\
                result = transform(op, [10, 20, 30])
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[11, 21, 31]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "op = lambda x: x * 10")
        nb_runner.run_all()
        assert "[100, 200, 300]" in nb_runner.get_output(3)
