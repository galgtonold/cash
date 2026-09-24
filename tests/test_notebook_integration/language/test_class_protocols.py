"""Dunder methods: repr and str, hashing and equality, ordering, callables and containers."""

import textwrap

import pytest


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


@pytest.mark.stress
class TestContainerDunders:
    """Test container protocol dunders."""

    def test_change_propagation_dunders(self, nb_runner):
        """Operator result propagation after change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Money:
                    def __init__(self, amount, currency='USD'):
                        self.amount = amount
                        self.currency = currency
                    def __add__(self, other):
                        if self.currency != other.currency:
                            raise ValueError("Currency mismatch")
                        return Money(self.amount + other.amount, self.currency)
                    def __repr__(self):
                        return f"{self.amount} {self.currency}"

                a = Money(100)
                b = Money(50)
            """),
                textwrap.dedent("""\
                total = a + b
                print(f"total={total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=150 USD" in nb_runner.get_output(2)

        # Change amount
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Money:
                def __init__(self, amount, currency='USD'):
                    self.amount = amount
                    self.currency = currency
                def __add__(self, other):
                    if self.currency != other.currency:
                        raise ValueError("Currency mismatch")
                    return Money(self.amount + other.amount, self.currency)
                def __repr__(self):
                    return f"{self.amount} {self.currency}"

            a = Money(200)
            b = Money(75)
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "total=275 USD" in nb_runner.get_output(2)


@pytest.mark.stress
class TestCustomContainers:
    """Test custom container classes."""

    def test_matrix_container(self, nb_runner):
        """Custom matrix with __getitem__ and __setitem__."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Matrix:
                    def __init__(self, rows, cols, fill=0):
                        self.rows = rows
                        self.cols = cols
                        self.data = [[fill] * cols for _ in range(rows)]

                    def __getitem__(self, key):
                        r, c = key
                        return self.data[r][c]

                    def __setitem__(self, key, value):
                        r, c = key
                        self.data[r][c] = value

                    def __repr__(self):
                        return f"Matrix({self.rows}x{self.cols})"

                m = Matrix(3, 3)
                for i in range(3):
                    m[i, i] = 1  # identity
            """),
                textwrap.dedent("""\
                diag = [m[i, i] for i in range(3)]
                off_diag = m[0, 1]
                print(f"m={m} diag={diag} off={off_diag}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Matrix(3x3)" in nb_runner.get_output(2)
        assert "diag=[1, 1, 1]" in nb_runner.get_output(2)
        assert "off=0" in nb_runner.get_output(2)

    def test_default_dict_like(self, nb_runner):
        """Custom defaultdict-like with factory across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class AutoDict(dict):
                    def __init__(self, factory):
                        super().__init__()
                        self.factory = factory

                    def __missing__(self, key):
                        self[key] = self.factory()
                        return self[key]

                word_counts = AutoDict(int)
                words = "the cat sat on the mat the cat".split()
                for w in words:
                    word_counts[w] += 1
            """),
                textwrap.dedent("""\
                sorted_counts = sorted(word_counts.items())
                print(f"counts={sorted_counts}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('the', 3)" in out
        assert "('cat', 2)" in out
