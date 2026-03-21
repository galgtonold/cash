"""
Batch 27: Descriptor, property, slots, dataclass, and protocol patterns.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestDescriptorPatterns:
    """Test caching with Python descriptors."""

    def test_property_decorator(self, nb_runner):
        """Class with @property across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Circle:
                    def __init__(self, radius):
                        self._radius = radius
                    
                    @property
                    def radius(self):
                        return self._radius
                    
                    @property
                    def area(self):
                        import math
                        return math.pi * self._radius ** 2
            """),
            "c = Circle(5)",
            textwrap.dedent("""\
                print(f"r={c.radius} a={c.area:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=5 a=78.54" in nb_runner.get_output(3)

    def test_property_change_class(self, nb_runner):
        """Change property logic → downstream updates."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Rect:
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                    
                    @property
                    def area(self):
                        return self.w * self.h
            """),
            "r = Rect(4, 5)",
            "print(r.area)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        # Change property to include perimeter
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class Rect:
                def __init__(self, w, h):
                    self.w = w
                    self.h = h
                
                @property
                def area(self):
                    return self.w * self.h
                
                @property
                def perimeter(self):
                    return 2 * (self.w + self.h)
        """))
        nb_runner.set_cell_source(3, "print(f'{r.area} {r.perimeter}')")
        nb_runner.run_all()
        assert "20 18" in nb_runner.get_output(3)


class TestDataclassPatterns:
    """Test caching with dataclasses."""

    def test_basic_dataclass(self, nb_runner):
        """Dataclass creation and usage across cells."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field",
            textwrap.dedent("""\
                @dataclass
                class Point:
                    x: float
                    y: float
                    
                    def distance(self):
                        return (self.x**2 + self.y**2)**0.5
            """),
            "p = Point(3.0, 4.0)",
            "print(f'd={p.distance():.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=5.0" in nb_runner.get_output(4)

    def test_dataclass_with_default_factory(self, nb_runner):
        """Dataclass with field(default_factory=...)."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field",
            textwrap.dedent("""\
                @dataclass
                class Config:
                    name: str
                    tags: list = field(default_factory=list)
                    meta: dict = field(default_factory=dict)
            """),
            textwrap.dedent("""\
                c = Config('test', ['a', 'b'], {'key': 'val'})
                print(f"{c.name} tags={c.tags} meta={c.meta}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "test" in output
        assert "['a', 'b']" in output

    def test_frozen_dataclass(self, nb_runner):
        """Frozen dataclass (immutable)."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            textwrap.dedent("""\
                @dataclass(frozen=True)
                class FrozenPoint:
                    x: int
                    y: int
            """),
            textwrap.dedent("""\
                p = FrozenPoint(1, 2)
                # Should be hashable
                d = {p: 'origin'}
                print(d[FrozenPoint(1, 2)])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "origin" in nb_runner.get_output(3)

    def test_dataclass_inheritance(self, nb_runner):
        """Dataclass inheritance across cells."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            textwrap.dedent("""\
                @dataclass
                class Base:
                    name: str
                    value: int
            """),
            textwrap.dedent("""\
                @dataclass
                class Extended(Base):
                    extra: str = 'default'
            """),
            textwrap.dedent("""\
                e = Extended('test', 42, 'custom')
                print(f"{e.name} {e.value} {e.extra}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "test 42 custom" in nb_runner.get_output(4)


class TestSlotsPatterns:
    """Test caching with __slots__ classes."""

    def test_slots_class(self, nb_runner):
        """Class with __slots__."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Compact:
                    __slots__ = ('x', 'y')
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
            """),
            "c = Compact(10, 20)",
            "print(f'{c.x} {c.y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10 20" in nb_runner.get_output(3)


class TestProtocolPatterns:
    """Test structural subtyping and duck typing."""

    def test_protocol_like_duck_typing(self, nb_runner):
        """Duck typing: different classes, same interface."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Dog:
                    def speak(self):
                        return "Woof"

                class Cat:
                    def speak(self):
                        return "Meow"
            """),
            textwrap.dedent("""\
                def make_noise(animal):
                    return animal.speak()
            """),
            textwrap.dedent("""\
                d = Dog()
                c = Cat()
                print(f"{make_noise(d)} {make_noise(c)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Woof Meow" in nb_runner.get_output(3)


class TestNamedTuplePatterns:
    """Test caching with named tuples."""

    def test_namedtuple_basic(self, nb_runner):
        """collections.namedtuple across cells."""
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Point = namedtuple('Point', ['x', 'y'])",
            textwrap.dedent("""\
                p = Point(3, 4)
                dist = (p.x**2 + p.y**2)**0.5
                print(f"{p} d={dist:.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Point(x=3, y=4)" in output
        assert "d=5.0" in output

    def test_typing_namedtuple(self, nb_runner):
        """typing.NamedTuple with type hints."""
        nb_runner.create_notebook([
            "from typing import NamedTuple",
            textwrap.dedent("""\
                class Employee(NamedTuple):
                    name: str
                    dept: str
                    salary: float
            """),
            textwrap.dedent("""\
                e = Employee('Alice', 'Eng', 120000)
                print(f"{e.name} in {e.dept}: ${e.salary:,.0f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Alice in Eng: $120,000" in output


class TestContextManagerPatterns:
    """Test caching with context managers."""

    def test_contextmanager_decorator(self, nb_runner):
        """contextlib.contextmanager used across cells."""
        nb_runner.create_notebook([
            "from contextlib import contextmanager",
            textwrap.dedent("""\
                @contextmanager
                def temp_value(container, key, value):
                    old = container.get(key)
                    container[key] = value
                    try:
                        yield container
                    finally:
                        if old is None:
                            del container[key]
                        else:
                            container[key] = old
            """),
            textwrap.dedent("""\
                config = {'mode': 'prod'}
                with temp_value(config, 'mode', 'test') as c:
                    inside = c['mode']
                outside = config['mode']
                print(f"inside={inside} outside={outside}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "inside=test outside=prod" in nb_runner.get_output(3)

    def test_custom_context_manager_class(self, nb_runner):
        """Custom __enter__/__exit__ context manager."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Timer:
                    def __enter__(self):
                        import time
                        self.start = time.time()
                        return self
                    def __exit__(self, *args):
                        import time
                        self.elapsed = time.time() - self.start
            """),
            textwrap.dedent("""\
                import time
                with Timer() as t:
                    time.sleep(0.01)
                print(f"elapsed={t.elapsed > 0}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "elapsed=True" in nb_runner.get_output(2)
