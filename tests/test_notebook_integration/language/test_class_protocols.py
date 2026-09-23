"""Dunder methods, hashing, equality, __slots__ and class/static methods."""

import textwrap

import pytest


# Interaction test: class with __call__ and callable protocol.
# Tests classes implementing __call__, callable checks,
# and cross-cell callable composition.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCallableProtocol:
    """Test callable protocol via __call__ across cells."""

    def test_callable_class(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define callable class
                "class Multiplier:\n    def __init__(self, factor):\n        self.factor = factor\n        self.call_count = 0\n    def __call__(self, value):\n        self.call_count += 1\n        return value * self.factor\n    def __repr__(self):\n        return f'Multiplier(x{self.factor})'\nprint('Multiplier defined')",
                # Cell 2: use as callable
                "double = Multiplier(2)\ntriple = Multiplier(3)\nresults = [double(5), triple(5), double(10), triple(10)]\nprint(f'results={results}')\nprint(f'double_calls={double.call_count}')\nprint(f'triple_calls={triple.call_count}')",
                # Cell 3: compose
                "composed = double(triple(7))\nprint(f'composed={composed}')\nprint(f'all_callable={all(callable(f) for f in [double, triple])}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "results=[10, 15, 20, 30]" in out2
        assert "double_calls=2" in out2
        assert "triple_calls=2" in out2
        out3 = nb_runner.get_output(3)
        assert "composed=42" in out3
        assert "all_callable=True" in out3

    def test_callable_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Adder:\n    def __init__(self, n):\n        self.n = n\n    def __call__(self, x):\n        return x + self.n\nprint('Adder defined')",
                "add5 = Adder(5)\nresult = add5(10)\nprint(f'result={result}')",
                "doubled = add5(result)\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
        assert "doubled=20" in nb_runner.get_output(3)

        # Edit adder value
        nb_runner.set_cell_source(2, "add5 = Adder(10)\nresult = add5(10)\nprint(f'result={result}')")
        nb_runner.run_cells([2, 3])
        assert "result=20" in nb_runner.get_output(2)
        assert "doubled=30" in nb_runner.get_output(3)

    def test_callable_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Squarer:\n    def __call__(self, x):\n        return x ** 2\nsq = Squarer()\nprint(f'callable={callable(sq)}')",
                "vals = [sq(i) for i in range(5)]\nprint(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "callable=True" in nb_runner.get_output(1)
        assert "vals=[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "vals=[0, 1, 4, 9, 16]" in nb_runner.get_output(2)


# class method / static method patterns with caching.
# Tests @classmethod, @staticmethod, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClassStaticMethod:
    """Test classmethod and staticmethod caching."""

    def test_classmethod_factory(self, nb_runner):
        """classmethod as factory with caching."""
        nb_runner.create_notebook(
            [
                "class Date:\n    def __init__(self, y, m, d):\n        self.y = y\n        self.m = m\n        self.d = d\n    @classmethod\n    def from_string(cls, s):\n        y, m, d = map(int, s.split('-'))\n        return cls(y, m, d)\n    def __str__(self):\n        return f'{self.y}/{self.m}/{self.d}'",
                "d = Date.from_string('2024-06-15')",
                "print(f'date={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "date=2024/6/15" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "date=2024/6/15" in out2

    def test_staticmethod_edit(self, nb_runner):
        """staticmethod with edit propagation."""
        nb_runner.create_notebook(
            [
                "class MathHelper:\n    @staticmethod\n    def clamp(val, lo, hi):\n        return max(lo, min(hi, val))",
                "val = 150",
                "result = MathHelper.clamp(val, 0, 100)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result=100" in out

        nb_runner.set_cell_source(2, "val = 50")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result=50" in out2

    def test_classmethod_counter(self, nb_runner):
        """classmethod tracking instance count."""
        nb_runner.create_notebook(
            [
                "class Widget:\n    _count = 0\n    def __init__(self, name):\n        self.name = name\n        Widget._count += 1\n    @classmethod\n    def get_count(cls):\n        return cls._count",
                "w1 = Widget('A')\nw2 = Widget('B')\nw3 = Widget('C')\ncount = Widget.get_count()",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=3" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "count=3" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClassMethodStaticMethod:
    """class method and static method patterns."""

    def test_classmethod_factory(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup cell",
                "class Date:\n    def __init__(self, y, m, d): self.y, self.m, self.d = y, m, d\n    @classmethod\n    def from_string(cls, s):\n        y, m, d = map(int, s.split('-'))\n        return cls(y, m, d)\n    def __str__(self): return f'{self.y}/{self.m}/{self.d}'\nd = Date.from_string('2024-01-15')\nprint(f'date={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "date=2024/1/15" in nb_runner.get_output(2)

    def test_staticmethod_util(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup cell",
                "class MathUtils:\n    @staticmethod\n    def is_prime(n):\n        if n < 2: return False\n        return all(n % i != 0 for i in range(2, int(n**0.5)+1))\nprimes = [n for n in range(20) if MathUtils.is_prime(n)]\nprint(f'primes={primes}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "primes=[2, 3, 5, 7, 11, 13, 17, 19]" in nb_runner.get_output(2)

    def test_classmethod_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup cell",
                "class Counter:\n    count = 0\n    @classmethod\n    def increment(cls): cls.count += 1\n    @classmethod\n    def get(cls): return cls.count\nCounter.increment()\nCounter.increment()\nprint(f'count={Counter.get()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "class Counter:\n    count = 0\n    @classmethod\n    def increment(cls): cls.count += 1\n    @classmethod\n    def get(cls): return cls.count\nfor _ in range(5): Counter.increment()\nprint(f'count={Counter.get()}')",
        )
        nb_runner.run_all()
        assert "count=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestContainerProtocol:
    """class __contains__, __len__, __getitem__ protocol."""

    def test_container_protocol(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Bag:\n    def __init__(self, items):\n        self._items = list(items)\n    def __contains__(self, item):\n        return item in self._items\n    def __len__(self):\n        return len(self._items)\n    def __getitem__(self, idx):\n        return self._items[idx]",
                "b = Bag([10, 20, 30])\nhas_20 = 20 in b\nlength = len(b)\nfirst = b[0]\nprint(f'has_20={has_20} length={length} first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "has_20=True length=3 first=10" in nb_runner.get_output(2)

    def test_container_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Stack:\n    def __init__(self):\n        self._data = []\n    def push(self, val):\n        self._data.append(val)\n    def __len__(self):\n        return len(self._data)\n    def __getitem__(self, idx):\n        return self._data[idx]",
                "s = Stack()\nfor v in [1, 2, 3]:\n    s.push(v)\nresult = f'{len(s)},{s[-1]}'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3,3" in nb_runner.get_output(2)
        # Edit to push different values
        nb_runner.set_cell_source(
            2,
            "s = Stack()\nfor v in [10, 20, 30, 40]:\n    s.push(v)\nresult = f'{len(s)},{s[-1]}'\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=4,40" in nb_runner.get_output(2)

    def test_bool_protocol(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class NonEmpty:\n    def __init__(self, items):\n        self.items = items\n    def __bool__(self):\n        return len(self.items) > 0",
                "a = NonEmpty([1])\nb = NonEmpty([])\nprint(f'a={bool(a)} b={bool(b)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=True b=False" in nb_runner.get_output(2)


# Custom __hash__/__eq__ interaction tests.
# Tests that objects with custom hashing/equality properly interact
# with cash's caching when their definitions or data change.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCustomHashEqInteraction:
    """Test custom __hash__/__eq__ with cache invalidation."""

    def test_hashable_object_edit(self, nb_runner):
        """Editing a class with custom __hash__ should invalidate downstream."""
        nb_runner.create_notebook(
            [
                (
                    "class Point:\n"
                    "    def __init__(self, x, y):\n"
                    "        self.x = x\n"
                    "        self.y = y\n"
                    "    def __hash__(self):\n"
                    "        return hash((self.x, self.y))\n"
                    "    def __eq__(self, other):\n"
                    "        return self.x == other.x and self.y == other.y\n"
                    "    def __repr__(self):\n"
                    "        return f'Point({self.x},{self.y})'"
                ),
                "p = Point(3, 4)",
                "s = {p, Point(1, 2)}\ncount = len(s)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=2" in out

        # Change to create duplicate (same hash)
        nb_runner.set_cell_source(2, "p = Point(1, 2)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=1" in out

    def test_dict_key_custom_hash_edit(self, nb_runner):
        """Editing objects used as dict keys with custom hash."""
        nb_runner.create_notebook(
            [
                (
                    "class Key:\n"
                    "    def __init__(self, name):\n"
                    "        self.name = name\n"
                    "    def __hash__(self):\n"
                    "        return hash(self.name)\n"
                    "    def __eq__(self, other):\n"
                    "        return self.name == other.name"
                ),
                "k = Key('alpha')\nmapping = {k: 100, Key('beta'): 200}",
                "val = mapping[Key('alpha')]",
                "print(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "val=100" in out

        nb_runner.set_cell_source(2, "k = Key('alpha')\nmapping = {k: 999, Key('beta'): 200}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "val=999" in out

    def test_frozen_dataclass_hash_edit(self, nb_runner):
        """Frozen dataclass (auto-hashing) edit should propagate."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                ("@dataclass(frozen=True)\nclass Config:\n    name: str\n    version: int"),
                "c1 = Config('app', 1)\nc2 = Config('app', 2)\nconfigs = {c1, c2}",
                "count = len(configs)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=2" in out

        # Make them the same
        nb_runner.set_cell_source(3, "c1 = Config('app', 1)\nc2 = Config('app', 1)\nconfigs = {c1, c2}")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=1" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDunderReprEq:
    """custom __repr__, __str__, __eq__ dunder methods."""

    def test_repr_str(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'\n    def __str__(self):\n        return f'({self.x}, {self.y})'",
                "p = Point(3, 4)\nrepr_s = repr(p)\nstr_s = str(p)\nprint(f'repr={repr_s} str={str_s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "repr=Point(3, 4)" in out
        assert "str=(3, 4)" in out

    def test_eq_hash_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Color:\n    def __init__(self, r, g, b):\n        self.r = r\n        self.g = g\n        self.b = b\n    def __eq__(self, other):\n        return (self.r, self.g, self.b) == (other.r, other.g, other.b)\n    def __hash__(self):\n        return hash((self.r, self.g, self.b))",
                "c1 = Color(255, 0, 0)\nc2 = Color(255, 0, 0)\nc3 = Color(0, 255, 0)\neq12 = c1 == c2\neq13 = c1 == c3\nprint(f'eq12={eq12} eq13={eq13}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "eq12=True eq13=False" in nb_runner.get_output(2)
        # Edit class
        nb_runner.set_cell_source(
            1,
            "class Color:\n    def __init__(self, r, g, b):\n        self.r = r\n        self.g = g\n        self.b = b\n    def __eq__(self, other):\n        return self.r == other.r\n    def __hash__(self):\n        return hash(self.r)",
        )
        nb_runner.set_cell_source(
            2, "c1 = Color(255, 0, 0)\nc2 = Color(255, 100, 200)\neq = c1 == c2\nprint(f'eq={eq}')"
        )
        nb_runner.run_all()
        assert "eq=True" in nb_runner.get_output(2)

    def test_lt_sort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price\n    def __lt__(self, other):\n        return self.price < other.price\n    def __repr__(self):\n        return f'{self.name}:{self.price}'",
                "items = [Item('b', 30), Item('a', 10), Item('c', 20)]\nsorted_items = sorted(items)\nprint(f'sorted={sorted_items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sorted=[a:10, c:20, b:30]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHashEquality:
    """hash() and equality protocol."""

    def test_hash_eq(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Point:\n    def __init__(self, x, y): self.x, self.y = x, y\n    def __hash__(self): return hash((self.x, self.y))\n    def __eq__(self, other): return (self.x, self.y) == (other.x, other.y)",
                "p1 = Point(1, 2)\np2 = Point(1, 2)\np3 = Point(3, 4)\nprint(f'eq12={p1 == p2} eq13={p1 == p3} hash_eq={hash(p1) == hash(p2)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eq12=True" in out
        assert "eq13=False" in out
        assert "hash_eq=True" in out

    def test_hashable_in_set(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Color:\n    def __init__(self, r, g, b): self.r, self.g, self.b = r, g, b\n    def __hash__(self): return hash((self.r, self.g, self.b))\n    def __eq__(self, other): return (self.r, self.g, self.b) == (other.r, other.g, other.b)",
                "s = {Color(255, 0, 0), Color(0, 255, 0), Color(255, 0, 0)}\nprint(f'count={len(s)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)

    def test_hash_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Key:\n    def __init__(self, val): self.val = val\n    def __hash__(self): return hash(self.val)\n    def __eq__(self, o): return self.val == o.val",
                "d = {Key(1): 'a', Key(2): 'b'}\nresult = d[Key(1)]\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=a" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = {Key(10): 'x', Key(20): 'y'}\nresult = d[Key(20)]\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=y" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHashEqualityDunder:
    """hash and equality dunder methods."""

    def test_hash_eq_in_set(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Card:\n    def __init__(self, rank, suit):\n        self.rank, self.suit = rank, suit\n    def __eq__(self, other):\n        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit\n    def __hash__(self):\n        return hash((self.rank, self.suit))\n    def __repr__(self):\n        return f'{self.rank}{self.suit}'\nc1 = Card('A', 'S')\nc2 = Card('A', 'S')\nc3 = Card('K', 'H')\nhand = {c1, c2, c3}\nprint(f'eq={c1 == c2} len={len(hand)} hand={sorted(str(c) for c in hand)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eq=True" in out
        assert "len=2" in out

    def test_comparable_objects(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Score:\n    def __init__(self, v): self.v = v\n    def __lt__(self, o): return self.v < o.v\n    def __eq__(self, o): return self.v == o.v\n    def __repr__(self): return f'S({self.v})'\nscores = [Score(5), Score(3), Score(8), Score(1)]\nordered = sorted(scores)\nprint(f'ordered={ordered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ordered=[S(1), S(3), S(5), S(8)]" in nb_runner.get_output(2)

    def test_hash_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Pt:\n    def __init__(self, x, y): self.x, self.y = x, y\n    def __hash__(self): return hash((self.x, self.y))\n    def __eq__(self, o): return (self.x, self.y) == (o.x, o.y)\ns = {Pt(1,2), Pt(1,2), Pt(3,4)}\nprint(f'len={len(s)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "class Pt:\n    def __init__(self, x, y): self.x, self.y = x, y\n    def __hash__(self): return hash((self.x, self.y))\n    def __eq__(self, o): return (self.x, self.y) == (o.x, o.y)\ns = {Pt(1,2), Pt(3,4), Pt(5,6)}\nprint(f'len={len(s)}')",
        )
        nb_runner.run_all()
        assert "len=3" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestReprStr:
    """class __repr__ and __str__ methods."""

    def test_repr_str(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Coin:\n    def __init__(self, name, value):\n        self.name = name\n        self.value = value\n    def __repr__(self): return f'Coin({self.name!r}, {self.value})'\n    def __str__(self): return f'{self.name}=${self.value}'",
                "c = Coin('quarter', 0.25)\nr = repr(c)\ns = str(c)\nprint(f'repr={r} str={s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "repr=Coin('quarter', 0.25)" in out
        assert "str=quarter=$0.25" in out

    def test_format_method(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Duration:\n    def __init__(self, seconds):\n        self.seconds = seconds\n    def __format__(self, spec):\n        if spec == 'hms':\n            h, rem = divmod(self.seconds, 3600)\n            m, s = divmod(rem, 60)\n            return f'{h}h{m}m{s}s'\n        return str(self.seconds)",
                "d = Duration(3661)\nprint(f'hms={d:hms} raw={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hms=1h1m1s" in nb_runner.get_output(2)
        assert "raw=3661" in nb_runner.get_output(2)

    def test_repr_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Tag:\n    def __init__(self, name): self.name = name\n    def __repr__(self): return f'Tag({self.name!r})'",
                "t = Tag('python')\nprint(f'tag={t!r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "tag=Tag('python')" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "t = Tag('rust')\nprint(f'tag={t!r}')")
        nb_runner.run_all()
        assert "tag=Tag('rust')" in nb_runner.get_output(2)


# Interaction test: class __slots__ with inheritance and memory optimization.
# Tests __slots__ classes with inheritance, MRO slot resolution,
# and cross-cell attribute access patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsInheritance:
    """Test __slots__ with inheritance across cells."""

    def test_slots_inheritance(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define base with slots
                "class Point2D:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'P2({self.x},{self.y})'\nclass Point3D(Point2D):\n    __slots__ = ('z',)\n    def __init__(self, x, y, z):\n        super().__init__(x, y)\n        self.z = z\n    def __repr__(self):\n        return f'P3({self.x},{self.y},{self.z})'\nprint('Point classes defined')",
                # Cell 2: create and use
                "p2 = Point2D(1, 2)\np3 = Point3D(3, 4, 5)\nprint(f'p2={p2}')\nprint(f'p3={p3}')\nprint(f'p2_slots={Point2D.__slots__}')\nprint(f'p3_slots={Point3D.__slots__}')",
                # Cell 3: compute with points
                "import math\ndist = math.sqrt((p3.x - p2.x)**2 + (p3.y - p2.y)**2)\nprint(f'dist_2d={dist:.2f}')\ndist_3d = math.sqrt(p3.x**2 + p3.y**2 + p3.z**2)\nprint(f'dist_origin={dist_3d:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "p2=P2(1,2)" in out2
        assert "p3=P3(3,4,5)" in out2
        assert "p2_slots=('x', 'y')" in out2
        assert "p3_slots=('z',)" in out2
        out3 = nb_runner.get_output(3)
        assert "dist_2d=2.83" in out3
        assert "dist_origin=7.07" in out3

    def test_slots_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def mag(self):\n        return (self.x**2 + self.y**2)**0.5\nprint('Vec defined')",
                "v = Vec(3, 4)\nprint(f'mag={v.mag()}')",
                "scaled_mag = v.mag() * 2\nprint(f'scaled={scaled_mag}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=5.0" in nb_runner.get_output(2)
        assert "scaled=10.0" in nb_runner.get_output(3)

        # Edit vector values
        nb_runner.set_cell_source(2, "v = Vec(5, 12)\nprint(f'mag={v.mag()}')")
        nb_runner.run_cells([2, 3])
        assert "mag=13.0" in nb_runner.get_output(2)
        assert "scaled=26.0" in nb_runner.get_output(3)

    def test_slots_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Coord:\n    __slots__ = ('lat', 'lon')\n    def __init__(self, lat, lon):\n        self.lat = lat\n        self.lon = lon\nprint('Coord defined')",
                "c = Coord(40.7128, -74.0060)\nhas_dict = hasattr(c, '__dict__')\nprint(f'lat={c.lat}')\nprint(f'has_dict={has_dict}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lat=40.7128" in nb_runner.get_output(2)
        assert "has_dict=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_dict=False" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsMemory:
    """class __slots__ for memory efficiency."""

    def test_slots_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Point:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
                "p = Point(3, 4)\nhas_dict = hasattr(p, '__dict__')\nhas_slots = hasattr(p, '__slots__')\nprint(f'x={p.x} y={p.y} has_dict={has_dict} has_slots={has_slots}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "x=3" in out
        assert "has_dict=False" in out
        assert "has_slots=True" in out

    def test_slots_no_extra_attr(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Pair:\n    __slots__ = ('a', 'b')\n    def __init__(self, a, b): self.a, self.b = a, b",
                "p = Pair(1, 2)\ntry:\n    p.c = 3\n    result = 'no_error'\nexcept AttributeError:\n    result = 'attr_error'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=attr_error" in nb_runner.get_output(2)

    def test_slots_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y): self.x, self.y = x, y\n    def mag_sq(self): return self.x**2 + self.y**2",
                "v = Vec(3, 4)\nprint(f'mag_sq={v.mag_sq()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag_sq=25" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "v = Vec(5, 12)\nprint(f'mag_sq={v.mag_sq()}')")
        nb_runner.run_all()
        assert "mag_sq=169" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsOptimization:
    """class slots optimization and attribute access."""

    def test_slots_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Point:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
                "p = Point(3, 4)\nresult = f'{p.x},{p.y}'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3,4" in nb_runner.get_output(2)

    def test_slots_edit_class(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def mag(self):\n        return (self.x**2 + self.y**2) ** 0.5",
                "v = Vec(3, 4)\nm = v.mag()\nprint(f'm={m}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "m=5.0" in nb_runner.get_output(2)
        # Edit to 3D
        nb_runner.set_cell_source(
            1,
            "class Vec:\n    __slots__ = ('x', 'y', 'z')\n    def __init__(self, x, y, z):\n        self.x = x\n        self.y = y\n        self.z = z\n    def mag(self):\n        return (self.x**2 + self.y**2 + self.z**2) ** 0.5",
        )
        nb_runner.set_cell_source(2, "v = Vec(1, 2, 2)\nm = v.mag()\nprint(f'm={m}')")
        nb_runner.run_all()
        assert "m=3.0" in nb_runner.get_output(2)

    def test_slots_no_dict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Tiny:\n    __slots__ = ('val',)\n    def __init__(self, v):\n        self.val = v",
                "t = Tiny(42)\nhas_dict = hasattr(t, '__dict__')\nprint(f'val={t.val} has_dict={has_dict}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=42 has_dict=False" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStaticClassMethods:
    """static methods and class methods."""

    def test_classmethod(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Date:\n    def __init__(self, y, m, d): self.y, self.m, self.d = y, m, d\n    @classmethod\n    def from_string(cls, s):\n        parts = s.split('-')\n        return cls(int(parts[0]), int(parts[1]), int(parts[2]))",
                "d = Date.from_string('2024-06-15')\nprint(f'y={d.y} m={d.m} d={d.d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y=2024" in nb_runner.get_output(2)
        assert "m=6" in nb_runner.get_output(2)
        assert "d=15" in nb_runner.get_output(2)

    def test_staticmethod(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Math:\n    @staticmethod\n    def clamp(val, lo, hi):\n        return max(lo, min(val, hi))",
                "r1 = Math.clamp(5, 0, 10)\nr2 = Math.clamp(-5, 0, 10)\nr3 = Math.clamp(15, 0, 10)\nprint(f'r1={r1} r2={r2} r3={r3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=5" in nb_runner.get_output(2)
        assert "r2=0" in nb_runner.get_output(2)
        assert "r3=10" in nb_runner.get_output(2)

    def test_classmethod_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Config:\n    def __init__(self, **kw): self.data = kw\n    @classmethod\n    def default(cls): return cls(mode='fast', debug=False)",
                'c = Config.default()\nprint(f\'mode={c.data["mode"]} debug={c.data["debug"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mode=fast" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            1,
            "class Config:\n    def __init__(self, **kw): self.data = kw\n    @classmethod\n    def default(cls): return cls(mode='slow', debug=True)",
        )
        nb_runner.run_all()
        assert "mode=slow" in nb_runner.get_output(2)
        assert "debug=True" in nb_runner.get_output(2)


# Interaction test: functools.total_ordering with rich comparison.
# Tests @total_ordering decorator for auto-generating comparison methods,
# sorting, and cross-cell usage with min/max.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsTotalOrdering:
    """Test functools.total_ordering across cells."""

    def test_total_ordering_comparisons(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define ordered class
                "from functools import total_ordering\n@total_ordering\nclass Temperature:\n    def __init__(self, celsius):\n        self.celsius = celsius\n    def __eq__(self, other):\n        return self.celsius == other.celsius\n    def __lt__(self, other):\n        return self.celsius < other.celsius\n    def __repr__(self):\n        return f'T({self.celsius})'\nprint('Temperature defined')",
                # Cell 2: create and compare
                "t1 = Temperature(20)\nt2 = Temperature(30)\nt3 = Temperature(20)\nprint(f'lt={t1 < t2}')\nprint(f'gt={t2 > t1}')\nprint(f'eq={t1 == t3}')\nprint(f'le={t1 <= t3}')\nprint(f'ge={t2 >= t1}')\nprint(f'ne={t1 != t2}')",
                # Cell 3: sort and min/max
                "temps = [Temperature(25), Temperature(10), Temperature(35), Temperature(15)]\nsorted_temps = sorted(temps)\nprint(f'sorted={sorted_temps}')\nprint(f'min={min(temps)}')\nprint(f'max={max(temps)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "lt=True" in out2
        assert "gt=True" in out2
        assert "eq=True" in out2
        assert "le=True" in out2
        assert "ge=True" in out2
        assert "ne=True" in out2
        out3 = nb_runner.get_output(3)
        assert "sorted=[T(10), T(15), T(25), T(35)]" in out3
        assert "min=T(10)" in out3
        assert "max=T(35)" in out3

    def test_total_ordering_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import total_ordering\n@total_ordering\nclass Score:\n    def __init__(self, val):\n        self.val = val\n    def __eq__(self, other):\n        return self.val == other.val\n    def __lt__(self, other):\n        return self.val < other.val\n    def __repr__(self):\n        return f'S({self.val})'\nprint('Score defined')",
                "scores = [Score(85), Score(92), Score(78)]\nbest = max(scores)\nprint(f'best={best}')",
                "ranking = sorted(scores, reverse=True)\nprint(f'rank={ranking}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=S(92)" in nb_runner.get_output(2)
        assert "rank=[S(92), S(85), S(78)]" in nb_runner.get_output(3)

        # Add a new score
        nb_runner.set_cell_source(
            2, "scores = [Score(85), Score(92), Score(78), Score(99)]\nbest = max(scores)\nprint(f'best={best}')"
        )
        nb_runner.run_cells([2, 3])
        assert "best=S(99)" in nb_runner.get_output(2)
        assert "rank=[S(99), S(92), S(85), S(78)]" in nb_runner.get_output(3)

    def test_total_ordering_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import total_ordering\n@total_ordering\nclass Version:\n    def __init__(self, major, minor):\n        self.major = major\n        self.minor = minor\n    def __eq__(self, other):\n        return (self.major, self.minor) == (other.major, other.minor)\n    def __lt__(self, other):\n        return (self.major, self.minor) < (other.major, other.minor)\n    def __repr__(self):\n        return f'v{self.major}.{self.minor}'\nprint('Version defined')",
                "versions = [Version(2, 1), Version(1, 9), Version(2, 0)]\nlatest = max(versions)\nprint(f'latest={latest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "latest=v2.1" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "latest=v2.1" in nb_runner.get_output(2)


# Slots and __slots__ interaction tests.
# Tests that editing classes with __slots__ and their instances
# properly invalidates downstream cells.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsInteraction:
    """Test __slots__ class patterns with cache invalidation."""

    def test_slots_class_instance_edit(self, nb_runner):
        """Editing a slots-based class instance should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Vector:\n"
                    "    __slots__ = ('x', 'y')\n"
                    "    def __init__(self, x, y):\n"
                    "        self.x = x\n"
                    "        self.y = y\n"
                    "    def magnitude(self):\n"
                    "        return (self.x**2 + self.y**2) ** 0.5"
                ),
                "v = Vector(3, 4)",
                "mag = round(v.magnitude(), 2)",
                "print(f'mag={mag}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mag=5.0" in out

        nb_runner.set_cell_source(2, "v = Vector(5, 12)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mag=13.0" in out

    def test_slots_class_definition_edit(self, nb_runner):
        """Editing the class definition with __slots__ should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Record:\n"
                    "    __slots__ = ('name', 'value')\n"
                    "    def __init__(self, name, value):\n"
                    "        self.name = name\n"
                    "        self.value = value\n"
                    "    def display(self):\n"
                    "        return f'{self.name}={self.value}'"
                ),
                "r = Record('temp', 25)",
                "text = r.display()",
                "print(f'text={text}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "text=temp=25" in out

        # Edit class to change display format
        nb_runner.set_cell_source(
            1,
            (
                "class Record:\n"
                "    __slots__ = ('name', 'value')\n"
                "    def __init__(self, name, value):\n"
                "        self.name = name\n"
                "        self.value = value\n"
                "    def display(self):\n"
                "        return f'{self.name}: {self.value}'"
            ),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "text=temp: 25" in out


# __slots__, memory optimization & class patterns — cash caching.
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
