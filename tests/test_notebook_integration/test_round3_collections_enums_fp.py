"""
Batch 15: Collections patterns, enum usage, protocol/structural typing,
__slots__, and complex comprehension patterns.

Tests how cash handles specialized collection types, enums across cells,
Protocol-based structural subtyping, __slots__ classes, and deeply nested
comprehensions.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress]


# ============================================================
# Test Group 1: Collections Module Patterns
# ============================================================

class TestCollectionsPatterns:
    """Test collections module usage across cells."""

    def test_counter_across_cells(self, nb_runner):
        """Counter created in one cell, queried in another."""
        nb_runner.create_notebook([
            "from collections import Counter",
            "words = 'the quick brown fox jumps over the lazy fox'.split()",
            textwrap.dedent("""\
                counts = Counter(words)
                print(counts.most_common(2))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "the" in output
        assert "fox" in output

    def test_defaultdict_pattern(self, nb_runner):
        """defaultdict with lambda factory."""
        nb_runner.create_notebook([
            "from collections import defaultdict",
            textwrap.dedent("""\
                groups = defaultdict(list)
                for item in [('a', 1), ('b', 2), ('a', 3), ('b', 4)]:
                    groups[item[0]].append(item[1])
                print(dict(groups))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "'a': [1, 3]" in output
        assert "'b': [2, 4]" in output

    def test_namedtuple_across_cells(self, nb_runner):
        """namedtuple defined in one cell, used in another."""
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Point = namedtuple('Point', ['x', 'y'])",
            textwrap.dedent("""\
                p = Point(3, 4)
                dist = (p.x**2 + p.y**2) ** 0.5
                print(f"{dist:.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "5.0" in nb_runner.get_output(3)

    def test_deque_operations(self, nb_runner):
        """deque with maxlen across cells."""
        nb_runner.create_notebook([
            "from collections import deque",
            "d = deque(maxlen=3)",
            textwrap.dedent("""\
                for i in range(5):
                    d.append(i)
                print(list(d))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 3, 4]" in nb_runner.get_output(3)

    def test_ordereddict_pattern(self, nb_runner):
        """OrderedDict insertion order across cells."""
        nb_runner.create_notebook([
            "from collections import OrderedDict",
            textwrap.dedent("""\
                od = OrderedDict()
                od['banana'] = 3
                od['apple'] = 1
                od['cherry'] = 2
            """),
            textwrap.dedent("""\
                keys = list(od.keys())
                print(keys)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['banana', 'apple', 'cherry']" in nb_runner.get_output(3)

    def test_chainmap_pattern(self, nb_runner):
        """ChainMap for layered configuration."""
        nb_runner.create_notebook([
            "from collections import ChainMap",
            textwrap.dedent("""\
                defaults = {'color': 'red', 'size': 10}
                overrides = {'color': 'blue'}
                config = ChainMap(overrides, defaults)
                print(config['color'], config['size'])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "blue 10" in nb_runner.get_output(2)


# ============================================================
# Test Group 2: Enum Patterns
# ============================================================

class TestEnumPatterns:
    """Test enum usage across cells."""

    def test_basic_enum(self, nb_runner):
        """Enum defined in one cell, used in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import Enum

                class Color(Enum):
                    RED = 1
                    GREEN = 2
                    BLUE = 3
            """),
            textwrap.dedent("""\
                c = Color.RED
                print(c.name, c.value)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "RED 1" in nb_runner.get_output(2)

    def test_enum_with_methods(self, nb_runner):
        """Enum with custom methods."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import Enum

                class Direction(Enum):
                    NORTH = (0, 1)
                    SOUTH = (0, -1)
                    EAST = (1, 0)
                    WEST = (-1, 0)

                    def move(self, x, y, steps=1):
                        dx, dy = self.value
                        return x + dx * steps, y + dy * steps
            """),
            textwrap.dedent("""\
                pos = (0, 0)
                pos = Direction.NORTH.move(*pos, steps=3)
                pos = Direction.EAST.move(*pos, steps=2)
                print(pos)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(2, 3)" in nb_runner.get_output(2)

    def test_int_enum(self, nb_runner):
        """IntEnum for numeric comparison."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import IntEnum

                class Priority(IntEnum):
                    LOW = 1
                    MEDIUM = 2
                    HIGH = 3
            """),
            textwrap.dedent("""\
                tasks = [Priority.HIGH, Priority.LOW, Priority.MEDIUM]
                sorted_tasks = sorted(tasks)
                print([t.name for t in sorted_tasks])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['LOW', 'MEDIUM', 'HIGH']" in nb_runner.get_output(2)

    def test_enum_change_invalidation(self, nb_runner):
        """Changing enum definition should invalidate downstream."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import Enum
                class Status(Enum):
                    ACTIVE = 'active'
                    INACTIVE = 'inactive'
            """),
            textwrap.dedent("""\
                s = Status.ACTIVE
                print(s.value)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "active" in nb_runner.get_output(2)

        # Change enum
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            from enum import Enum
            class Status(Enum):
                ACTIVE = 'enabled'
                INACTIVE = 'disabled'
        """))
        nb_runner.run_all()
        assert "enabled" in nb_runner.get_output(2)


# ============================================================
# Test Group 3: __slots__ Classes
# ============================================================

class TestSlotsClasses:
    """Test __slots__ class patterns."""

    def test_basic_slots(self, nb_runner):
        """Class with __slots__."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Point:
                    __slots__ = ('x', 'y')
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
            """),
            textwrap.dedent("""\
                p = Point(1, 2)
                print(p.x, p.y)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "1 2" in nb_runner.get_output(2)

    def test_slots_no_dict(self, nb_runner):
        """__slots__ class has no __dict__."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Compact:
                    __slots__ = ('value',)
                    def __init__(self, v):
                        self.value = v
            """),
            textwrap.dedent("""\
                obj = Compact(42)
                has_dict = hasattr(obj, '__dict__')
                print(f"value={obj.value}, has_dict={has_dict}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "value=42" in output
        assert "has_dict=False" in output

    def test_slots_inheritance(self, nb_runner):
        """__slots__ with inheritance."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    __slots__ = ('x',)
                    def __init__(self, x):
                        self.x = x

                class Child(Base):
                    __slots__ = ('y',)
                    def __init__(self, x, y):
                        super().__init__(x)
                        self.y = y
            """),
            textwrap.dedent("""\
                c = Child(10, 20)
                print(c.x + c.y)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(2)


# ============================================================
# Test Group 4: Complex Comprehension Patterns
# ============================================================

class TestComplexComprehensions:
    """Test deeply nested and complex comprehension patterns."""

    def test_nested_list_comprehension(self, nb_runner):
        """Nested list comprehension with cross-cell dependency."""
        nb_runner.create_notebook([
            "matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
            textwrap.dedent("""\
                flat = [x for row in matrix for x in row if x % 2 == 0]
                print(flat)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8]" in nb_runner.get_output(2)

    def test_dict_comprehension_complex(self, nb_runner):
        """Dict comprehension with conditional and cross-cell data."""
        nb_runner.create_notebook([
            "names = ['Alice', 'Bob', 'Charlie', 'David']",
            "scores = [85, 92, 78, 95]",
            textwrap.dedent("""\
                passing = {n: s for n, s in zip(names, scores) if s >= 80}
                print(passing)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'Alice': 85" in output
        assert "'Bob': 92" in output
        assert "'David': 95" in output
        assert "Charlie" not in output

    def test_set_comprehension(self, nb_runner):
        """Set comprehension with cross-cell dependency."""
        nb_runner.create_notebook([
            "data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]",
            textwrap.dedent("""\
                unique_doubled = {x * 2 for x in data}
                print(sorted(unique_doubled))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8]" in nb_runner.get_output(2)

    def test_generator_expression_materialized(self, nb_runner):
        """Generator expression consumed across cells."""
        nb_runner.create_notebook([
            "numbers = range(1, 11)",
            textwrap.dedent("""\
                gen = (x**2 for x in numbers if x % 3 == 0)
                squares = list(gen)
                print(squares)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[9, 36, 81]" in nb_runner.get_output(2)

    def test_nested_dict_comprehension(self, nb_runner):
        """Nested dict/list comprehension."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']",
            textwrap.dedent("""\
                nested = {k: [i * ord(k) for i in range(3)] for k in keys}
                print(nested['a'])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        # ord('a')=97, so [0, 97, 194]
        assert "0" in output
        assert "97" in output


# ============================================================
# Test Group 5: Functional Programming Patterns
# ============================================================

class TestFunctionalPatterns:
    """Test functional programming patterns across cells."""

    def test_partial_application(self, nb_runner):
        """functools.partial across cells."""
        nb_runner.create_notebook([
            "from functools import partial",
            textwrap.dedent("""\
                def power(base, exp):
                    return base ** exp

                square = partial(power, exp=2)
                cube = partial(power, exp=3)
            """),
            textwrap.dedent("""\
                print(square(5), cube(3))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25 27" in nb_runner.get_output(3)

    def test_lru_cache_decorator(self, nb_runner):
        """lru_cache decorated function across cells."""
        nb_runner.create_notebook([
            "from functools import lru_cache",
            textwrap.dedent("""\
                @lru_cache(maxsize=128)
                def fib(n):
                    if n < 2:
                        return n
                    return fib(n-1) + fib(n-2)
            """),
            textwrap.dedent("""\
                result = fib(30)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "832040" in nb_runner.get_output(3)

    def test_operator_module(self, nb_runner):
        """Using operator module functions as higher-order function args."""
        nb_runner.create_notebook([
            "import operator",
            "from functools import reduce",
            "numbers = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                product = reduce(operator.mul, numbers)
                print(product)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "120" in nb_runner.get_output(4)

    def test_compose_functions(self, nb_runner):
        """Function composition pattern."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def compose(*funcs):
                    def composed(x):
                        for f in reversed(funcs):
                            x = f(x)
                        return x
                    return composed
            """),
            textwrap.dedent("""\
                double = lambda x: x * 2
                add_one = lambda x: x + 1
                square = lambda x: x ** 2
            """),
            textwrap.dedent("""\
                transform = compose(square, add_one, double)
                result = transform(3)  # double(3)=6, add_one(6)=7, square(7)=49
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "49" in nb_runner.get_output(3)

    def test_higher_order_function_change(self, nb_runner):
        """Changing a higher-order function's component should invalidate."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def apply_twice(f, x):
                    return f(f(x))
            """),
            textwrap.dedent("""\
                def increment(x):
                    return x + 1
            """),
            textwrap.dedent("""\
                result = apply_twice(increment, 5)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, textwrap.dedent("""\
            def increment(x):
                return x + 10
        """))
        nb_runner.run_all()
        assert "25" in nb_runner.get_output(3)  # increment(increment(5)) = 5+10+10=25
