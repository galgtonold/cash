"""
Batch 30: Global state, singleton, registry, and mutable default argument patterns.
Tests tricky Python patterns that interact with caching in subtle ways.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestSingletonPatterns:
    """Test singleton-like patterns across cells."""

    def test_module_level_singleton(self, nb_runner):
        """Module-level singleton pattern."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Config:
                    _instance = None
                    def __new__(cls):
                        if cls._instance is None:
                            cls._instance = super().__new__(cls)
                            cls._instance.debug = False
                        return cls._instance
            """),
            textwrap.dedent("""\
                c1 = Config()
                c1.debug = True
                c2 = Config()
                print(f"same={c1 is c2} debug={c2.debug}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "same=True debug=True" in nb_runner.get_output(2)

    def test_registry_pattern(self, nb_runner):
        """Registry pattern: register handlers across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                _registry = {}
                def register(name):
                    def decorator(fn):
                        _registry[name] = fn
                        return fn
                    return decorator
            """),
            textwrap.dedent("""\
                @register('add')
                def add(a, b):
                    return a + b

                @register('mul')
                def mul(a, b):
                    return a * b
            """),
            textwrap.dedent("""\
                result = _registry['add'](3, 4)
                print(f"add={result} registered={sorted(_registry.keys())}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "add=7" in output
        assert "['add', 'mul']" in output


class TestMutableDefaultArguments:
    """Test caching with mutable default arguments."""

    def test_mutable_default_list(self, nb_runner):
        """Classic mutable default argument gotcha."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def append_to(item, lst=None):
                    if lst is None:
                        lst = []
                    lst.append(item)
                    return lst
            """),
            textwrap.dedent("""\
                r1 = append_to(1)
                r2 = append_to(2)
                r3 = append_to(3, [10, 20])
                print(f"r1={r1} r2={r2} r3={r3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "r1=[1]" in output
        assert "r2=[2]" in output
        assert "r3=[10, 20, 3]" in output


class TestGlobalStateCrossCell:
    """Test global state management across cells."""

    def test_global_counter(self, nb_runner):
        """Global counter incremented across cells."""
        nb_runner.create_notebook([
            "counter = 0",
            textwrap.dedent("""\
                def increment(n=1):
                    global counter
                    counter += n
                    return counter
            """),
            textwrap.dedent("""\
                increment(5)
                increment(3)
                print(f"counter={counter}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "counter=8" in nb_runner.get_output(3)

    def test_class_variable_shared_state(self, nb_runner):
        """Class variable shared between instances across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Counter:
                    count = 0
                    def __init__(self):
                        Counter.count += 1
                    def get_count(self):
                        return Counter.count
            """),
            textwrap.dedent("""\
                a = Counter()
                b = Counter()
                c = Counter()
                print(f"count={c.get_count()}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(2)


class TestDunderMethodPatterns:
    """Test caching with special dunder methods."""

    def test_repr_and_str(self, nb_runner):
        """Custom __repr__ and __str__ across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Item:
                    def __init__(self, name, price):
                        self.name = name
                        self.price = price
                    def __repr__(self):
                        return f"Item({self.name!r}, {self.price})"
                    def __str__(self):
                        return f"{self.name}: ${self.price:.2f}"
            """),
            textwrap.dedent("""\
                item = Item("Widget", 19.99)
                print(f"repr={repr(item)} str={item}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "repr=Item('Widget', 19.99)" in output
        assert "str=Widget: $19.99" in output

    def test_comparison_operators(self, nb_runner):
        """Custom comparison operators across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Version:
                    def __init__(self, major, minor, patch):
                        self.major = major
                        self.minor = minor
                        self.patch = patch
                    def __lt__(self, other):
                        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
                    def __eq__(self, other):
                        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
                    def __repr__(self):
                        return f"{self.major}.{self.minor}.{self.patch}"
            """),
            textwrap.dedent("""\
                versions = [Version(2, 0, 0), Version(1, 5, 3), Version(1, 5, 0), Version(3, 0, 1)]
                sorted_v = sorted(versions)
                print([str(v) for v in sorted_v])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "1.5.0" in output
        assert "3.0.1" in output

    def test_container_protocol(self, nb_runner):
        """__getitem__, __len__, __contains__ across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Matrix:
                    def __init__(self, data):
                        self._data = data
                    def __getitem__(self, idx):
                        row, col = idx
                        return self._data[row][col]
                    def __len__(self):
                        return len(self._data)
                    def __contains__(self, val):
                        return any(val in row for row in self._data)
            """),
            textwrap.dedent("""\
                m = Matrix([[1, 2], [3, 4], [5, 6]])
                print(f"[1,0]={m[1,0]} len={len(m)} has_4={4 in m} has_9={9 in m}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1,0]=3 len=3 has_4=True has_9=False" in nb_runner.get_output(2)

    def test_arithmetic_operators(self, nb_runner):
        """Custom __add__, __mul__ operators."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Vec2:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
                    def __add__(self, other):
                        return Vec2(self.x + other.x, self.y + other.y)
                    def __mul__(self, scalar):
                        return Vec2(self.x * scalar, self.y * scalar)
                    def __repr__(self):
                        return f"Vec2({self.x}, {self.y})"
            """),
            textwrap.dedent("""\
                a = Vec2(1, 2)
                b = Vec2(3, 4)
                c = a + b
                d = a * 3
                print(f"c={c} d={d}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c=Vec2(4, 6) d=Vec2(3, 6)" in nb_runner.get_output(2)
