"""Batch 75: __slots__, memory optimization & class patterns — cash caching."""
import textwrap
import pytest


@pytest.mark.stress
class TestSlotsPatterns:
    """Test __slots__ class patterns across cells."""

    def test_slots_class(self, nb_runner):
        """Class with __slots__ across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Point:
                    __slots__ = ('x', 'y')
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
                    def __repr__(self):
                        return f"Point({self.x}, {self.y})"

                p = Point(3, 4)
                print(f"point={p}")
                has_dict = hasattr(p, '__dict__')
                print(f"has_dict={has_dict}")
            """),
            textwrap.dedent("""\
                import math
                dist = math.sqrt(p.x ** 2 + p.y ** 2)
                print(f"distance={dist}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "point=Point(3, 4)" in nb_runner.get_output(1)
        assert "has_dict=False" in nb_runner.get_output(1)
        assert "distance=5.0" in nb_runner.get_output(2)

    def test_slots_inheritance(self, nb_runner):
        """Slots with inheritance across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    __slots__ = ('x',)
                    def __init__(self, x):
                        self.x = x

                class Derived(Base):
                    __slots__ = ('y',)
                    def __init__(self, x, y):
                        super().__init__(x)
                        self.y = y
                    def __repr__(self):
                        return f"D({self.x}, {self.y})"

                d = Derived(10, 20)
                print(f"derived={d}")
            """),
            textwrap.dedent("""\
                total = d.x + d.y
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "derived=D(10, 20)" in nb_runner.get_output(1)
        assert "total=30" in nb_runner.get_output(2)

    def test_slots_many_instances(self, nb_runner):
        """Many slotted instances across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Record:
                    __slots__ = ('id', 'value')
                    def __init__(self, id, value):
                        self.id = id
                        self.value = value

                records = [Record(i, i * 10) for i in range(1000)]
                print(f"created={len(records)}")
            """),
            textwrap.dedent("""\
                total = sum(r.value for r in records)
                print(f"total={total}")
                first5 = [(r.id, r.value) for r in records[:5]]
                print(f"first5={first5}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "created=1000" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "total=4995000" in out2
        assert "(0, 0)" in out2


@pytest.mark.stress
class TestMemoryOptimization:
    """Test memory optimization patterns."""

    def test_intern_strings(self, nb_runner):
        """sys.intern for string optimization across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=100 unique=5" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "cat_0" in out2
        assert "20" in out2  # Each category appears 20 times

    def test_slots_change_propagation(self, nb_runner):
        """Slots class — value extraction propagates changes."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "level=20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            multiplier = 5
        """))
        nb_runner.run_cells([1, 2])
        assert "level=50" in nb_runner.get_output(2)
