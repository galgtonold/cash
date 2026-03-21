"""
Round 3 - Batch 12: Advanced edge cases and uncommon Python patterns.

Tests focusing on:
1. Global/nonlocal keyword interactions
2. Type annotation patterns (no runtime effect)
3. Property decorators and descriptors
4. Dataclass patterns
5. Complex unpacking patterns
6. Generator/iterator patterns across cells
7. Complex string operations (multiline, raw, bytes)
8. Chained method calls
9. Boolean logic chains
10. Default argument patterns
"""

import pytest



pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestGlobalNonlocalPatterns:
    """Tests for global and nonlocal keyword interactions."""

    @pytest.mark.core
    def test_global_variable_in_function(self, nb_runner):
        """Function using global keyword should track the variable."""
        nb_runner.create_notebook([
            "counter = 0",
            "def increment():\n    global counter\n    counter += 1\n    return counter",
            "r1 = increment()\nr2 = increment()",
            "print(f'counter={counter} r1={r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "counter=2" in output
        assert "r1=1" in output
        assert "r2=2" in output

    @pytest.mark.core
    def test_nonlocal_in_nested_function(self, nb_runner):
        """Nonlocal keyword in nested function should work."""
        nb_runner.create_notebook([
            "def make_counter(start=0):\n    count = start\n    def inc():\n        nonlocal count\n        count += 1\n        return count\n    return inc",
            "counter = make_counter(10)",
            "v1 = counter()\nv2 = counter()\nv3 = counter()",
            "print(f'v1={v1} v2={v2} v3={v3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "v1=11" in output
        assert "v2=12" in output
        assert "v3=13" in output


class TestPropertyAndDescriptorPatterns:
    """Tests for property decorators and descriptor protocol."""

    @pytest.mark.core
    def test_class_with_property(self, nb_runner):
        """Property decorator should work within caching."""
        nb_runner.create_notebook([
            "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n    @property\n    def area(self):\n        import math\n        return math.pi * self._radius ** 2\n    @property\n    def radius(self):\n        return self._radius\n    @radius.setter\n    def radius(self, value):\n        self._radius = value",
            "c = Circle(5)",
            "area1 = round(c.area, 2)",
            "print(f'area1={area1}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "area1=78.54" in output

        # Change class
        nb_runner.set_cell_source(1, "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n    @property\n    def area(self):\n        import math\n        return math.pi * self._radius ** 2 * 2\n    @property\n    def radius(self):\n        return self._radius\n    @radius.setter\n    def radius(self, value):\n        self._radius = value")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "area1=157.08" in output2


class TestDataclassPatterns:
    """Tests for dataclass usage patterns."""

    @pytest.mark.core
    def test_basic_dataclass(self, nb_runner):
        """Dataclass definition and usage."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass\nclass Point:\n    x: float\n    y: float\n    \n    def distance(self):\n        return (self.x**2 + self.y**2)**0.5",
            "p = Point(3.0, 4.0)",
            "d = p.distance()",
            "print(f'd={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "d=5.0" in output

    @pytest.mark.core
    def test_dataclass_with_default_factory(self, nb_runner):
        """Dataclass with default_factory."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field",
            "@dataclass\nclass Config:\n    name: str = 'default'\n    tags: list = field(default_factory=list)\n    options: dict = field(default_factory=dict)",
            "c1 = Config('test', ['a', 'b'], {'key': 'val'})",
            "c2 = Config()",
            "print(f'c1={c1.name} c2={c2.name} tags={c1.tags}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "c1=test" in output
        assert "c2=default" in output
        assert "tags=['a', 'b']" in output


class TestComplexUnpackingPatterns:
    """Tests for complex unpacking and assignment patterns."""

    @pytest.mark.core
    def test_nested_tuple_unpacking(self, nb_runner):
        """Nested tuple unpacking."""
        nb_runner.create_notebook([
            "data = ((1, 2), (3, 4), (5, 6))",
            "results = []\nfor (a, b) in data:\n    results.append(a + b)",
            "total = sum(results)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=21" in output  # 3+7+11

    @pytest.mark.core
    def test_dict_unpacking_merge(self, nb_runner):
        """Dict unpacking with ** operator."""
        nb_runner.create_notebook([
            "base = {'a': 1, 'b': 2}",
            "override = {'b': 20, 'c': 30}",
            "merged = {**base, **override}",
            "print(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "'a': 1" in output
        assert "'b': 20" in output
        assert "'c': 30" in output

    @pytest.mark.core
    def test_extended_unpacking_with_star(self, nb_runner):
        """Extended unpacking with * operator."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6, 7]",
            "first, *middle, last = data",
            "print(f'first={first} middle={middle} last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "first=1" in output
        assert "middle=[2, 3, 4, 5, 6]" in output
        assert "last=7" in output


class TestGeneratorIteratorPatterns:
    """Tests for generator and iterator patterns."""

    @pytest.mark.core
    def test_generator_function_across_cells(self, nb_runner):
        """Generator function defined in one cell, consumed in another."""
        nb_runner.create_notebook([
            "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
            "fib_list = list(fibonacci(10))",
            "total = sum(fib_list)",
            "print(f'total={total} last={fib_list[-1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=88" in output
        assert "last=34" in output

    @pytest.mark.core
    def test_itertools_chain(self, nb_runner):
        """Using itertools across cells."""
        nb_runner.create_notebook([
            "from itertools import chain, repeat, islice",
            "a = [1, 2, 3]\nb = [4, 5, 6]",
            "combined = list(chain(a, b))",
            "repeated = list(islice(repeat(42), 5))",
            "total = sum(combined) + sum(repeated)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "total=231" in output  # 21 + 210


class TestChainedMethodCalls:
    """Tests for chained method call patterns."""

    @pytest.mark.core
    def test_pandas_method_chaining(self, nb_runner):
        """Pandas method chaining pattern."""
        nb_runner.create_notebook([
            "import pandas as pd",
            "df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],\n    'value': [10, 20, 30, 40, 50]})",
            "result = (df\n    .groupby('name')['value']\n    .sum()\n    .sort_values(ascending=False)\n    .head(2))",
            "top_name = result.index[0]\ntop_val = result.iloc[0]",
            "print(f'top={top_name}:{top_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "top=" in output

    @pytest.mark.core
    def test_string_method_chaining(self, nb_runner):
        """String method chaining."""
        nb_runner.create_notebook([
            "raw = '  Hello, World!  Python 3.14  '",
            "cleaned = raw.strip().lower().replace(',', '').replace('!', '')",
            "words = cleaned.split()",
            "result = '-'.join(words)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=hello-world-python-3.14" in output


class TestBooleanLogicChains:
    """Tests for complex boolean logic across cells."""

    @pytest.mark.core
    def test_boolean_chain_across_cells(self, nb_runner):
        """Boolean conditions computed across cells."""
        nb_runner.create_notebook([
            "age = 25\nincome = 50000\ncredit_score = 720",
            "is_adult = age >= 18\nhas_income = income > 30000\ngood_credit = credit_score >= 700",
            "approved = is_adult and has_income and good_credit",
            "print(f'approved={approved}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "approved=True" in output

        # Change one condition to fail
        nb_runner.set_cell_source(1, "age = 16\nincome = 50000\ncredit_score = 720")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "approved=False" in output2

    @pytest.mark.core
    def test_any_all_patterns(self, nb_runner):
        """Using any() and all() built-in functions."""
        nb_runner.create_notebook([
            "scores = [85, 90, 78, 95, 88]",
            "all_passing = all(s >= 70 for s in scores)\nany_excellent = any(s >= 95 for s in scores)\nfailing = [s for s in scores if s < 80]",
            "print(f'all_passing={all_passing} any_excellent={any_excellent} failing={failing}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "all_passing=True" in output
        assert "any_excellent=True" in output
        assert "failing=[78]" in output


class TestDefaultArgumentPatterns:
    """Tests for default argument and mutable default patterns."""

    @pytest.mark.core
    def test_function_with_default_args(self, nb_runner):
        """Function with default arguments."""
        nb_runner.create_notebook([
            "def greet(name, greeting='Hello', exclaim=True):\n    msg = f'{greeting}, {name}'\n    return msg + '!' if exclaim else msg",
            "r1 = greet('World')\nr2 = greet('Cash', 'Hi', False)",
            "print(f'r1={r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "r1=Hello, World!" in output
        assert "r2=Hi, Cash" in output

    @pytest.mark.core
    def test_kwargs_pattern(self, nb_runner):
        """Function with **kwargs."""
        nb_runner.create_notebook([
            "def build_config(**kwargs):\n    return {k: v for k, v in kwargs.items() if v is not None}",
            "config = build_config(name='test', value=42, debug=None, verbose=True)",
            "print(f'config={config}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "'name': 'test'" in output
        assert "'value': 42" in output
        assert "'verbose': True" in output
        assert "debug" not in output


class TestMultilineStringPatterns:
    """Tests for multiline string and byte patterns."""

    @pytest.mark.core
    def test_multiline_string_formatting(self, nb_runner):
        """Multiline strings with formatting."""
        nb_runner.create_notebook([
            "name = 'Cash'\nversion = '2.0'",
            "template = f'''Project: {name}\nVersion: {version}\nStatus: Active'''",
            "lines = template.split('\\n')\nline_count = len(lines)",
            "print(f'line_count={line_count} first={lines[0]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "line_count=3" in output
        assert "first=Project: Cash" in output

    @pytest.mark.core
    def test_raw_string_pattern(self, nb_runner):
        """Raw strings (useful for regex, paths)."""
        nb_runner.create_notebook([
            "import re",
            "pattern = r'\\d{3}-\\d{3}-\\d{4}'",
            "text = 'Call 123-456-7890 or 098-765-4321'",
            "matches = re.findall(pattern, text)\ncount = len(matches)",
            "print(f'count={count} first={matches[0]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "count=2" in output
        assert "first=123-456-7890" in output


class TestComplexAssignmentPatterns:
    """Tests for augmented and complex assignment patterns."""

    @pytest.mark.core
    def test_augmented_assignment_operators(self, nb_runner):
        """All augmented assignment operators."""
        nb_runner.create_notebook([
            "x = 100",
            "x += 10\nx -= 5\nx *= 2\nx //= 3",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "x=70" in output  # ((100+10-5)*2)//3 = 210//3 = 70

    @pytest.mark.core
    def test_conditional_expression_assignment(self, nb_runner):
        """Conditional expressions in assignments."""
        nb_runner.create_notebook([
            "values = [3, 1, 4, 1, 5, 9, 2, 6]",
            "maximum = max(values)\nminimum = min(values)\nrange_val = maximum - minimum",
            "category = 'wide' if range_val > 5 else 'narrow'",
            "print(f'range={range_val} category={category}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "range=8" in output
        assert "category=wide" in output
