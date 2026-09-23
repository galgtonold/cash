"""Generators, send/throw, yield from and custom iterators across cells."""

import textwrap

import pytest


# Generator, iterator, and functional patterns — yield, send/throw,
# itertools, functools, closures, higher-order functions.
@pytest.mark.integration
@pytest.mark.stress
class TestGeneratorPatterns:
    """Test caching with generator functions and itertools."""

    def test_generator_consumed_to_list(self, nb_runner):
        """Generator consumed to list should cache."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def gen_range(n):
                    for i in range(n):
                        yield i * i
            """),
                textwrap.dedent("""\
                result = list(gen_range(5))
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run - should use cache
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_generator_expression(self, nb_runner):
        """Generator expression consumed to sum."""
        nb_runner.create_notebook(
            [
                "values = [1, 2, 3, 4, 5]",
                textwrap.dedent("""\
                total = sum(x**2 for x in values)
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "55" in nb_runner.get_output(2)

    def test_generator_function_change_propagation(self, nb_runner):
        """Change generator function → re-run consumer."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def fib(n):
                    a, b = 0, 1
                    for _ in range(n):
                        yield a
                        a, b = b, a + b
            """),
                textwrap.dedent("""\
                result = list(fib(6))
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 1, 1, 2, 3, 5]" in nb_runner.get_output(2)

        # Change to start from 1, 1
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def fib(n):
                a, b = 1, 1
                for _ in range(n):
                    yield a
                    a, b = b, a + b
        """),
        )
        nb_runner.run_all()
        assert "[1, 1, 2, 3, 5, 8]" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestClosurePatterns:
    """Test caching with closures and higher-order functions."""

    def test_higher_order_map_filter(self, nb_runner):
        """Higher-order functions: map + filter."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                textwrap.dedent("""\
                evens = list(filter(lambda x: x % 2 == 0, data))
                squared = list(map(lambda x: x**2, evens))
                print(squared)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[4, 16, 36, 64, 100]" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestFunctoolsPatterns:
    """Test caching with functools utilities."""

    def test_functools_lru_cache(self, nb_runner):
        """@lru_cache decorator on a function."""
        nb_runner.create_notebook(
            [
                "from functools import lru_cache",
                textwrap.dedent("""\
                @lru_cache(maxsize=128)
                def expensive(n):
                    return sum(range(n))
            """),
                textwrap.dedent("""\
                result = expensive(1000)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "499500" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCustomIterator:
    """custom iterator protocol (__iter__, __next__)."""

    def test_range_iterator(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class CountDown:\n    def __init__(self, start):\n        self.start = start\n    def __iter__(self):\n        self.current = self.start\n        return self\n    def __next__(self):\n        if self.current <= 0:\n            raise StopIteration\n        val = self.current\n        self.current -= 1\n        return val",
                "result = list(CountDown(5))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[5, 4, 3, 2, 1]" in nb_runner.get_output(2)

    def test_iterator_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Repeat:\n    def __init__(self, val, times):\n        self.val = val\n        self.times = times\n    def __iter__(self):\n        self.count = 0\n        return self\n    def __next__(self):\n        if self.count >= self.times:\n            raise StopIteration\n        self.count += 1\n        return self.val",
                "result = list(Repeat('x', 3))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['x', 'x', 'x']" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(
            1,
            "class Repeat:\n    def __init__(self, val, times):\n        self.val = val\n        self.times = times\n    def __iter__(self):\n        self.count = 0\n        return self\n    def __next__(self):\n        if self.count >= self.times:\n            raise StopIteration\n        self.count += 1\n        return self.val * self.count",
        )
        nb_runner.run_all()
        assert "result=['x', 'xx', 'xxx']" in nb_runner.get_output(2)

    def test_iter_for_loop(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Fibonacci:\n    def __init__(self, limit):\n        self.limit = limit\n    def __iter__(self):\n        a, b = 0, 1\n        while a < self.limit:\n            yield a\n            a, b = b, a + b",
                "fibs = list(Fibonacci(20))\nprint(f'fibs={fibs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fibs=[0, 1, 1, 2, 3, 5, 8, 13]" in nb_runner.get_output(2)


# generator/iterator protocol patterns with caching.
# Tests generator functions, iter/next, StopIteration, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestGeneratorIterator:
    """Test generator and iterator protocol caching."""

    def test_generator_function(self, nb_runner):
        """Generator function producing values with caching."""
        nb_runner.create_notebook(
            [
                "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
                "n = 8",
                "result = list(fib(n))\nprint(f'fib={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "fib=[0, 1, 1, 2, 3, 5, 8, 13]" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "fib=[0, 1, 1, 2, 3, 5, 8, 13]" in out2

    def test_generator_edit_count(self, nb_runner):
        """Edit generator count parameter, verify output changes."""
        nb_runner.create_notebook(
            [
                "def squares(n):\n    for i in range(n):\n        yield i ** 2",
                "count = 5",
                "result = list(squares(count))\nprint(f'squares={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "squares=[0, 1, 4, 9, 16]" in out

        nb_runner.set_cell_source(2, "count = 3")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "squares=[0, 1, 4]" in out2

    def test_generator_expression(self, nb_runner):
        """Generator expression with sum, verify caching."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "total = sum(x**2 for x in data)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total=55" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "total=55" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestGeneratorSendClose:
    """generator send and close protocol."""

    def test_generator_send(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def accumulator():\n    total = 0\n    while True:\n        val = yield total\n        if val is None: break\n        total += val",
                "gen = accumulator()\nnext(gen)\ngen.send(10)\ngen.send(20)\nresult = gen.send(30)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=60" in nb_runner.get_output(2)

    def test_yield_from(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def inner():\n    yield 1\n    yield 2\ndef outer():\n    yield 0\n    yield from inner()\n    yield 3",
                "result = list(outer())\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[0, 1, 2, 3]" in nb_runner.get_output(2)

    def test_generator_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def counter(start, step):\n    val = start\n    while True:\n        yield val\n        val += step",
                "gen = counter(0, 5)\nfirst5 = [next(gen) for _ in range(5)]\nprint(f'first5={first5}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first5=[0, 5, 10, 15, 20]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "gen = counter(100, 10)\nfirst5 = [next(gen) for _ in range(5)]\nprint(f'first5={first5}')"
        )
        nb_runner.run_all()
        assert "first5=[100, 110, 120, 130, 140]" in nb_runner.get_output(2)


# Interaction test: generator with send() and throw().
# Tests generator coroutine-like patterns with send(), throw(),
# and close(), verifying cross-cell generator state.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestGeneratorSendThrow:
    """Test generator send and throw across cells."""

    def test_generator_send(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define generator with send
                "def accumulator():\n    total = 0\n    while True:\n        value = yield total\n        if value is None:\n            break\n        total += value\nprint('accumulator defined')",
                # Cell 2: use send
                "gen = accumulator()\ncurrent = next(gen)  # prime\nprint(f'start={current}')\ncurrent = gen.send(10)\nprint(f'after_10={current}')\ncurrent = gen.send(20)\nprint(f'after_20={current}')\ncurrent = gen.send(5)\nprint(f'after_5={current}')",
                # Cell 3: use result
                "final = current\nprint(f'final={final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "start=0" in out2
        assert "after_10=10" in out2
        assert "after_20=30" in out2
        assert "after_5=35" in out2
        out3 = nb_runner.get_output(3)
        assert "final=35" in out3

    def test_generator_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def running_avg():\n    total = 0\n    count = 0\n    avg = 0\n    while True:\n        value = yield avg\n        if value is None:\n            break\n        total += value\n        count += 1\n        avg = total / count\nprint('running_avg defined')",
                "gen = running_avg()\nnext(gen)\ngen.send(10)\ngen.send(20)\nresult = gen.send(30)\nprint(f'avg={result}')",
                "rounded = round(result, 1)\nprint(f'rounded={rounded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=20.0" in nb_runner.get_output(2)
        assert "rounded=20.0" in nb_runner.get_output(3)

        # Change values
        nb_runner.set_cell_source(
            2, "gen = running_avg()\nnext(gen)\ngen.send(100)\nresult = gen.send(200)\nprint(f'avg={result}')"
        )
        nb_runner.run_cells([2, 3])
        assert "avg=150.0" in nb_runner.get_output(2)
        assert "rounded=150.0" in nb_runner.get_output(3)

    def test_generator_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def counter_gen(start=0):\n    n = start\n    while True:\n        reset = yield n\n        if reset is not None:\n            n = reset\n        else:\n            n += 1\nprint('counter_gen defined')",
                "g = counter_gen(10)\nv1 = next(g)\nv2 = next(g)\nv3 = g.send(100)\nv4 = next(g)\nprint(f'vals={[v1, v2, v3, v4]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=[10, 11, 100, 101]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "vals=[10, 11, 100, 101]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestGeneratorYieldFrom:
    """generator expressions and yield from."""

    def test_generator_expression(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = range(1, 11)",
                "squares_sum = sum(x**2 for x in data)\neven_sum = sum(x for x in data if x % 2 == 0)\nprint(f'squares_sum={squares_sum} even_sum={even_sum}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "squares_sum=385" in out
        assert "even_sum=30" in out

    def test_yield_from(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "def flatten(nested):\n    for item in nested:\n        if isinstance(item, list):\n            yield from flatten(item)\n        else:\n            yield item\ndata = [1, [2, 3], [4, [5, 6]], 7]\nresult = list(flatten(data))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[1, 2, 3, 4, 5, 6, 7]" in nb_runner.get_output(2)

    def test_generator_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "n = 5",
                "result = sum(i for i in range(n))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "n = 10")
        nb_runner.run_all()
        assert "result=45" in nb_runner.get_output(2)


# Iterator and protocol interaction tests.
#
# Tests editing cells with custom iterators, context managers,
# and protocol-based patterns.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestIteratorProtocolEdits:
    """Editing custom iterators and protocol patterns."""

    def test_edit_custom_range_iterator(self, nb_runner):
        """Edit a custom range-like iterator."""
        nb_runner.create_notebook(
            [
                "class MyRange:\n    def __init__(self, start, stop):\n        self.start = start\n        self.stop = stop\n    def __iter__(self):\n        current = self.start\n        while current < self.stop:\n            yield current\n            current += 1",
                "items = list(MyRange(0, 5))\nprint(f'items = {items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change step to 2
        nb_runner.set_cell_source(
            1,
            "class MyRange:\n    def __init__(self, start, stop):\n        self.start = start\n        self.stop = stop\n    def __iter__(self):\n        current = self.start\n        while current < self.stop:\n            yield current\n            current += 2",
        )
        nb_runner.run_all()
        assert "items = [0, 2, 4]" in nb_runner.get_output(2)

    def test_edit_container_with_len(self, nb_runner):
        """Edit a custom container with __len__ and __getitem__."""
        nb_runner.create_notebook(
            [
                "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def __len__(self):\n        return len(self._items)\n    def __repr__(self):\n        return f'Stack({self._items})'",
                "s = Stack()\ns.push(1)\ns.push(2)\ns.push(3)\nprint(f'len={len(s)} repr={s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3" in nb_runner.get_output(2)
        assert "Stack([1, 2, 3])" in nb_runner.get_output(2)

        # Edit to add only 2 items
        nb_runner.set_cell_source(2, "s = Stack()\ns.push(10)\ns.push(20)\nprint(f'len={len(s)} repr={s}')")
        nb_runner.run_all()
        assert "len=2" in nb_runner.get_output(2)
        assert "Stack([10, 20])" in nb_runner.get_output(2)

    def test_edit_callable_class(self, nb_runner):
        """Edit a callable class (__call__)."""
        nb_runner.create_notebook(
            [
                "class Transformer:\n    def __init__(self, scale):\n        self.scale = scale\n    def __call__(self, x):\n        return x * self.scale",
                "t = Transformer(3)\nresult = t(7)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(2)

        # Edit scale
        nb_runner.set_cell_source(2, "t = Transformer(5)\nresult = t(7)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 35" in nb_runner.get_output(2)


# Generator and iterator pipeline patterns.
#
# Tests generator functions, chaining, and edit propagation.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestGeneratorPipeline:
    """Generator pipeline patterns with edits."""

    def test_chained_generators(self, nb_runner):
        """Edit first generator in chain, final result updates."""
        nb_runner.create_notebook(
            [
                "def source(n):\n    for i in range(1, n + 1):\n        yield i",
                "def transform(gen):\n    for x in gen:\n        yield x * 10",
                "result = list(transform(source(4)))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [10, 20, 30, 40]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def source(n):\n    for i in range(1, n + 1):\n        yield i * 2",
        )
        nb_runner.run_all()
        assert "result = [20, 40, 60, 80]" in nb_runner.get_output(3)

    def test_generator_expression_edit(self, nb_runner):
        """Edit data fed into generator expression."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "squares = list(x**2 for x in data if x > 2)\nprint(f'squares = {squares}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squares = [9, 16, 25]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "squares = [100, 400, 900]" in nb_runner.get_output(2)


# advanced generators: send(), throw(), close(), yield from.
@pytest.mark.stress
@pytest.mark.integration
class TestGeneratorProtocol:
    """Generator send/throw/close protocol."""

    def test_generator_pipeline(self, nb_runner):
        """Coroutine-style generator pipeline."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def producer(n):
                    for i in range(n):
                        yield i * i

                def filterer(source, pred):
                    for item in source:
                        if pred(item):
                            yield item

                def mapper(source, fn):
                    for item in source:
                        yield fn(item)

                pipe = mapper(filterer(producer(10), lambda x: x % 2 == 0), lambda x: x + 1)
                output = list(pipe)
            """),
                "print(f'output={output}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        # 0,1,4,9,16,25,36,49,64,81 → even: 0,4,16,36,64 → +1: 1,5,17,37,65
        assert "1" in out
        assert "5" in out
        assert "17" in out

    def test_generator_propagation(self, nb_runner):
        """Generator that depends on upstream variable, change propagation."""
        nb_runner.create_notebook(
            [
                "multiplier = 2",
                textwrap.dedent("""\
                def scaled_range(n, scale):
                    for i in range(n):
                        yield i * scale

                collected = list(scaled_range(5, multiplier))
            """),
                "print(f'collected={collected}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 2, 4, 6, 8]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "multiplier = 10")
        nb_runner.run_cells([1, 2, 3])
        assert "[0, 10, 20, 30, 40]" in nb_runner.get_output(3)


# Iterator & custom container patterns — __iter__, __getitem__, __contains__.
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
