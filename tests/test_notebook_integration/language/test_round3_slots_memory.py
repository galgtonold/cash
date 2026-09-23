"""__slots__, memory optimization & class patterns — cash caching."""

import textwrap

import pytest


@pytest.mark.stress
class TestMemoryOptimization:
    """Test memory optimization patterns."""

    def test_intern_strings(self, nb_runner):
        """sys.intern for string optimization across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import sys

                # Create interned strings
                categories = [sys.intern(f"cat_{i % 5}") for i in range(100)]
                unique = set(categories)
                print(f"total={len(categories)} unique={len(unique)}")
            """),
                textwrap.dedent("""\
                from collections import Counter
                counts = Counter(categories)
                print(f"counts={dict(sorted(counts.items()))}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=100 unique=5" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "cat_0" in out2
        assert "20" in out2  # Each category appears 20 times

    def test_slots_change_propagation(self, nb_runner):
        """Slots class — value extraction propagates changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                multiplier = 2
            """),
                textwrap.dedent("""\
                class Config:
                    __slots__ = ('debug', 'level')
                    def __init__(self, debug, level):
                        self.debug = debug
                        self.level = level

                cfg = Config(True, multiplier * 10)
                level_val = cfg.level
                print(f"level={level_val}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "level=20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            multiplier = 5
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "level=50" in nb_runner.get_output(2)
