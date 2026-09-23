"""Editing a cell that defines a generator, context manager, comprehension or helper."""

import pytest

pytestmark = [pytest.mark.stress]


# Comprehension interaction tests.
#
# Tests editing list, dict, set, and generator comprehensions
# and verifying cache invalidation and recomputation.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestListComprehensionEdits:
    """List comprehension edits."""

    def test_edit_comprehension_expression(self, nb_runner):
        """Edit the expression in a list comprehension."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "result = [x * 2 for x in data]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change expression
        nb_runner.set_cell_source(2, "result = [x ** 2 for x in data]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_edit_comprehension_filter(self, nb_runner):
        """Edit the filter condition in a list comprehension."""
        nb_runner.create_notebook(
            [
                "nums = list(range(10))",
                "evens = [x for x in nums if x % 2 == 0]\nprint(f'evens = {evens}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change filter to odds
        nb_runner.set_cell_source(2, "evens = [x for x in nums if x % 2 == 1]\nprint(f'evens = {evens}')")
        nb_runner.run_all()
        assert "evens = [1, 3, 5, 7, 9]" in nb_runner.get_output(2)

    def test_edit_comprehension_source(self, nb_runner):
        """Edit the source data of a comprehension."""
        nb_runner.create_notebook(
            [
                "src = [10, 20, 30]  # source data",
                "doubled = [x * 2 for x in src]\nprint(f'doubled = {doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled = [20, 40, 60]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "src = [1, 2, 3]  # source data smaller")
        nb_runner.run_all()
        assert "doubled = [2, 4, 6]" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDictComprehensionEdits:
    """Dict comprehension edits."""

    def test_edit_dict_comprehension_value(self, nb_runner):
        """Edit the value expression in a dict comprehension."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']",
                "mapping = {k: len(k) for k in keys}\nprint(f'mapping = {mapping}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "mapping = {k: k.upper() for k in keys}\nprint(f'mapping = {mapping}')")
        nb_runner.run_all()
        assert "'a': 'A'" in nb_runner.get_output(2)

    def test_nested_comprehension_edit(self, nb_runner):
        """Edit a nested comprehension."""
        nb_runner.create_notebook(
            [
                "matrix = [[1, 2], [3, 4]]",
                "flat = [x for row in matrix for x in row]\nprint(f'flat = {flat}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat = [1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to transform
        nb_runner.set_cell_source(
            2,
            "flat = [x * 10 for row in matrix for x in row]\nprint(f'flat = {flat}')",
        )
        nb_runner.run_all()
        assert "flat = [10, 20, 30, 40]" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestSetComprehensionEdits:
    """Set comprehension edits."""

    def test_edit_set_comprehension(self, nb_runner):
        """Edit a set comprehension expression."""
        nb_runner.create_notebook(
            [
                "words = ['hello', 'world', 'hello', 'python']",
                "lengths = {len(w) for w in words}\nprint(f'lengths = {sorted(lengths)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lengths = [5, 6]" in nb_runner.get_output(2)

        # Change to first chars
        nb_runner.set_cell_source(2, "lengths = {w[0] for w in words}\nprint(f'lengths = {sorted(lengths)}')")
        nb_runner.run_all()
        assert "'h'" in nb_runner.get_output(2)
        assert "'p'" in nb_runner.get_output(2)
        assert "'w'" in nb_runner.get_output(2)


# Generator, iterator, and lazy evaluation interaction tests.
#
# Tests where generators and iterators are created in cells,
# consumed downstream, and cell edits affect the generation logic.
class TestGeneratorEdits:
    """Generator function edits."""

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_generator_function(self, nb_runner):
        """Edit generator function, verify consumer updates."""
        nb_runner.create_notebook(
            [
                "def gen_range(n):\n    for i in range(n):\n        yield i * 2",
                "result = list(gen_range(5))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Edit generator to yield squares
        nb_runner.set_cell_source(1, "def gen_range(n):\n    for i in range(n):\n        yield i ** 2")
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_generator_param(self, nb_runner):
        """Edit parameter passed to generator."""
        nb_runner.create_notebook(
            [
                "count = 3",
                "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
                "fibs = list(fib(count))\nprint(f'fibs = {fibs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fibs = [0, 1, 1]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "count = 8")
        nb_runner.run_all()
        assert "fibs = [0, 1, 1, 2, 3, 5, 8, 13]" in nb_runner.get_output(3)

    # Generator and iterator interaction tests.
    #
    # Tests editing generator functions, iterator protocols,
    # and lazy evaluation patterns.
    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_generator_yield(self, nb_runner):
        """Edit what a generator yields."""
        nb_runner.create_notebook(
            [
                "def gen_nums(n):\n    for i in range(n):\n        yield i",
                "result = list(gen_nums(5))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to yield squares
        nb_runner.set_cell_source(
            1,
            "def gen_nums(n):\n    for i in range(n):\n        yield i ** 2",
        )
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_generator_filter(self, nb_runner):
        """Edit the filter condition in a generator."""
        nb_runner.create_notebook(
            [
                "def even_gen(n):\n    for i in range(n):\n        if i % 2 == 0:\n            yield i",
                "result = list(even_gen(10))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change to odd
        nb_runner.set_cell_source(
            1,
            "def even_gen(n):\n    for i in range(n):\n        if i % 2 == 1:\n            yield i",
        )
        nb_runner.run_all()
        assert "result = [1, 3, 5, 7, 9]" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_generator_range(self, nb_runner):
        """Edit the range of a generator call."""
        nb_runner.create_notebook(
            [
                "def countdown(n):\n    while n > 0:\n        yield n\n        n -= 1",
                "result = list(countdown(3))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [3, 2, 1]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = list(countdown(6))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [6, 5, 4, 3, 2, 1]" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestMapFilterEdits:
    """Map/filter patterns with edits."""

    def test_edit_map_function(self, nb_runner):
        """Edit function used in map."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "mapped = list(map(lambda x: x * 2, data))",
                "total = sum(mapped)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "mapped = list(map(lambda x: x ** 2, data))")
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(3)

    def test_edit_filter_predicate(self, nb_runner):
        """Edit filter predicate."""
        nb_runner.create_notebook(
            [
                "nums = list(range(20))",
                "filtered = list(filter(lambda x: x % 2 == 0, nums))",
                "count = len(filtered)\nprint(f'count = {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "filtered = list(filter(lambda x: x % 5 == 0, nums))")
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(3)

    def test_chain_map_filter_edit(self, nb_runner):
        """Chain map then filter, edit map."""
        nb_runner.create_notebook(
            [
                "raw = list(range(1, 11))",
                "doubled = [x * 2 for x in raw]",
                "big = [x for x in doubled if x > 10]",
                "total = sum(big)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # doubled = [2,4,6,8,10,12,14,16,18,20], big = [12,14,16,18,20] -> 80
        assert "total = 80" in nb_runner.get_output(4)

        # Change map to triple
        nb_runner.set_cell_source(2, "doubled = [x * 3 for x in raw]")
        nb_runner.run_all()
        # tripled = [3,6,9,12,15,18,21,24,27,30], big = [12,15,18,21,24,27,30] -> 147
        assert "total = 147" in nb_runner.get_output(4)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestIteratorProtocol:
    """Iterator protocol with edits."""

    def test_edit_iterator_class(self, nb_runner):
        """Edit an iterator class __next__ method."""
        nb_runner.create_notebook(
            [
                "class Counter:\n    def __init__(self, n):\n        self.n = n\n        self.i = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.i >= self.n:\n            raise StopIteration\n        val = self.i\n        self.i += 1\n        return val",
                "result = list(Counter(4))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3]" in nb_runner.get_output(2)

        # Change to yield squares
        nb_runner.set_cell_source(
            1,
            "class Counter:\n    def __init__(self, n):\n        self.n = n\n        self.i = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.i >= self.n:\n            raise StopIteration\n        val = self.i ** 2\n        self.i += 1\n        return val",
        )
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9]" in nb_runner.get_output(2)

    def test_generator_expression_edit(self, nb_runner):
        """Edit a generator expression."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]  # source data for gen",
                "total = sum(x * 2 for x in data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

        # Change to cubed
        nb_runner.set_cell_source(2, "total = sum(x ** 3 for x in data)\nprint(f'total = {total}')")
        nb_runner.run_all()
        assert "total = 225" in nb_runner.get_output(2)


# Numeric precision and math interaction tests.
#
# Tests with floating point, integer overflow, precision changes,
# and mathematical operations combined with cell edits.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestFloatingPointEdits:
    """Floating point operations with edits."""

    def test_edit_precision(self, nb_runner):
        """Edit precision of rounding."""
        nb_runner.create_notebook(
            [
                "value = 3.141592653589793",
                "rounded = round(value, 2)\nprint(f'rounded = {rounded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "rounded = 3.14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "rounded = round(value, 4)\nprint(f'rounded = {rounded}')")
        nb_runner.run_all()
        assert "rounded = 3.1416" in nb_runner.get_output(2)

    def test_edit_math_operation(self, nb_runner):
        """Edit mathematical operation."""
        nb_runner.create_notebook(
            [
                "import math\nx = 16",
                "result = math.sqrt(x)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = math.log2(x)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(2)

    def test_accumulate_with_precision(self, nb_runner):
        """Accumulation with float precision."""
        nb_runner.create_notebook(
            [
                "values = [0.1] * 10",
                "total = sum(values)\nprint(f'total = {round(total, 1)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 1.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "values = [0.1] * 100")
        nb_runner.run_all()
        assert "total = 10.0" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestLargeNumberEdits:
    """Large number operations with edits."""

    def test_edit_exponent(self, nb_runner):
        """Edit exponentiation."""
        nb_runner.create_notebook(
            [
                "base = 2\nexp = 10",
                "result = base ** exp\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1024" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 2\nexp = 20")
        nb_runner.run_all()
        assert "result = 1048576" in nb_runner.get_output(2)

    def test_factorial_edit(self, nb_runner):
        """Edit factorial input."""
        nb_runner.create_notebook(
            [
                "import math\nn = 5",
                "result = math.factorial(n)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "import math\nn = 10")
        nb_runner.run_all()
        assert "result = 3628800" in nb_runner.get_output(2)


# String operations and formatting interaction tests.
#
# Tests where users perform string operations across cells,
# edit string content and formatting, and verify caching
# handles string changes correctly.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestStringEdits:
    """String manipulation with cell edits."""

    def test_edit_format_string(self, nb_runner):
        """Edit the format string itself."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "msg = f'The answer is {x}'\nprint(msg)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "The answer is 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "msg = f'Value: {x} (hex: {hex(x)})'\nprint(msg)")
        nb_runner.run_all()
        assert "Value: 42 (hex: 0x2a)" in nb_runner.get_output(2)

    def test_string_concatenation_chain(self, nb_runner):
        """Chain of string concatenation, edit source."""
        nb_runner.create_notebook(
            [
                "first = 'Hello'",
                "second = first + ' World'",
                "third = second + '!'\nprint(third)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World!" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "first = 'Goodbye'")
        nb_runner.run_all()
        assert "Goodbye World!" in nb_runner.get_output(3)

    def test_string_method_chain_edit(self, nb_runner):
        """String methods, edit the method call."""
        nb_runner.create_notebook(
            [
                "text = '  Hello World  '",
                "processed = text.strip()\nprint(f'|{processed}|')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "|Hello World|" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "processed = text.strip().upper()\nprint(f'|{processed}|')")
        nb_runner.run_all()
        assert "|HELLO WORLD|" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestStringParsingEdits:
    """String parsing patterns with cell edits."""

    def test_split_and_join_edit_delimiter(self, nb_runner):
        """Split/join with delimiter change."""
        nb_runner.create_notebook(
            [
                "raw = 'a,b,c,d'",
                "parts = raw.split(',')\nresult = '-'.join(parts)\nprint(result)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a-b-c-d" in nb_runner.get_output(2)

        # Change join delimiter
        nb_runner.set_cell_source(2, "parts = raw.split(',')\nresult = ' | '.join(parts)\nprint(result)")
        nb_runner.run_all()
        assert "a | b | c | d" in nb_runner.get_output(2)

    def test_regex_pattern_edit(self, nb_runner):
        """Regex pattern change."""
        nb_runner.create_notebook(
            [
                "import re\ntext = 'abc 123 def 456'",
                "nums = re.findall(r'\\d+', text)\nprint(f'nums = {nums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "nums = ['123', '456']" in nb_runner.get_output(2)

        # Change to match words
        nb_runner.set_cell_source(2, "words = re.findall(r'[a-z]+', text)\nprint(f'words = {words}')")
        nb_runner.run_all()
        assert "words = ['abc', 'def']" in nb_runner.get_output(2)


# Exception handling code interaction tests.
#
# Tests where try/except blocks are edited, error paths change,
# and caching handles exception-related code modifications.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestTryExceptEdits:
    """Edit try/except blocks."""

    def test_edit_try_body(self, nb_runner):
        """Edit the try body."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "try:\n    result = data[0]\nexcept IndexError:\n    result = -1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(2)

        # Edit to access out of bounds
        nb_runner.set_cell_source(
            2,
            "try:\n    result = data[10]\nexcept IndexError:\n    result = -1\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

    def test_edit_except_handler(self, nb_runner):
        """Edit the except handler."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "try:\n    result = 100 / x\nexcept ZeroDivisionError:\n    result = 0\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0" in nb_runner.get_output(2)

        # Edit to handle differently
        nb_runner.set_cell_source(
            2,
            "try:\n    result = 100 / x\nexcept ZeroDivisionError:\n    result = -999\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = -999" in nb_runner.get_output(2)

    def test_fix_value_to_avoid_exception(self, nb_runner):
        """Fix the value so exception doesn't trigger."""
        nb_runner.create_notebook(
            [
                "divisor = 0",
                "try:\n    result = 100 / divisor\nexcept ZeroDivisionError:\n    result = -1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

        # Fix divisor
        nb_runner.set_cell_source(1, "divisor = 4")
        nb_runner.run_all()
        assert "result = 25.0" in nb_runner.get_output(2)


class TestContextManagerEdits:
    """Context manager patterns with edits."""

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_context_body(self, nb_runner, tmp_path):
        """Edit code inside context manager."""
        fpath = tmp_path / "test.txt"
        fpath.write_text("hello world")
        fpath_str = str(fpath).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"path = '{fpath_str}'",
                "with open(path) as f:\n    content = f.read()\nresult = len(content)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        # Edit to get word count instead
        nb_runner.set_cell_source(
            2,
            "with open(path) as f:\n    content = f.read()\nresult = len(content.split())\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 2" in nb_runner.get_output(2)

    # Context manager pattern edits.
    #
    # Tests custom context managers with edits.
    @pytest.mark.timeout(90)
    def test_context_manager_class_edit(self, nb_runner):
        """Edit context manager class."""
        nb_runner.create_notebook(
            [
                "class Timer:\n    def __init__(self, label):\n        self.label = label\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass\n    def report(self):\n        return f'{self.label}: done'",
                "with Timer('task1') as t:\n    result = t.report()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = task1: done" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Timer:\n    def __init__(self, label):\n        self.label = label\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass\n    def report(self):\n        return f'[{self.label}] complete'",
        )
        nb_runner.run_all()
        assert "result = [task1] complete" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_contextmanager_decorator_edit(self, nb_runner):
        """Edit contextlib-based context manager."""
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef managed(name):\n    yield f'resource:{name}'",
                "with managed('db') as r:\n    val = r\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = resource:db" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from contextlib import contextmanager\n@contextmanager\ndef managed(name):\n    yield f'conn:{name}:active'",
        )
        nb_runner.run_all()
        assert "val = conn:db:active" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_with_statement_usage_edit(self, nb_runner):
        """Edit the with statement usage, keep manager same."""
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef scope(label):\n    yield label.upper()",
                "with scope('alpha') as s:\n    result = s\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ALPHA" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "with scope('beta') as s:\n    result = s\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = BETA" in nb_runner.get_output(2)


# Complex real-world simulation: data science pipeline.
#
# Full end-to-end data science workflow: load data, clean,
# feature engineer, model (simple), evaluate — with edits
# at each stage.
@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestDataSciencePipeline:
    """Full data science pipeline simulation."""

    def test_full_pipeline_edit_source(self, nb_runner, tmp_path):
        """Full pipeline, edit source data."""
        csv = tmp_path / "dataset.csv"
        csv.write_text("feature,target\n1,10\n2,20\n3,30\n4,40\n5,50\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "# Clean\ndf_clean = df.dropna()",
                "# Feature engineering\ndf_clean = df_clean.copy()\ndf_clean['feature_sq'] = df_clean['feature'] ** 2",
                "# Simple model: linear average\nmean_target = df_clean['target'].mean()\nprint(f'mean = {mean_target}')",
                "# Evaluate\nresiduals = [(t - mean_target) for t in df_clean['target']]\nrmse = (sum(r**2 for r in residuals) / len(residuals)) ** 0.5\nprint(f'rmse = {round(rmse, 2)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 30.0" in nb_runner.get_output(4)
        assert "rmse = " in nb_runner.get_output(5)

        # Edit source data
        csv.write_text("feature,target\n10,100\n20,200\n30,300\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 200.0" in nb_runner.get_output(4)

    def test_full_pipeline_edit_feature_engineering(self, nb_runner, tmp_path):
        """Edit feature engineering step."""
        csv = tmp_path / "data2.csv"
        csv.write_text("x,y\n1,2\n2,4\n3,6\n4,8\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "# Feature\nfeature_sum = df['x'].sum()\nprint(f'feature_sum = {feature_sum}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "feature_sum = 10" in nb_runner.get_output(2)

        # Change to use y column
        nb_runner.set_cell_source(
            2,
            "# Feature v2\nfeature_sum = df['y'].sum()\nprint(f'feature_sum = {feature_sum}')",
        )
        nb_runner.run_all()
        assert "feature_sum = 20" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestETLPipeline:
    """Extract-Transform-Load pipeline simulation."""

    def test_etl_edit_transform(self, nb_runner):
        """ETL pipeline, edit transform step."""
        nb_runner.create_notebook(
            [
                "# Extract\nraw = [{'name': 'Alice', 'score': 85}, {'name': 'Bob', 'score': 92}]",
                "# Transform\ntransformed = [{'name': r['name'], 'grade': 'A' if r['score'] >= 90 else 'B'} for r in raw]",
                "# Load (print)\nfor t in transformed:\n    print(f\"{t['name']}: {t['grade']}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Alice: B" in output
        assert "Bob: A" in output

        # Edit transform — lower A threshold
        nb_runner.set_cell_source(
            2,
            "# Transform v2\ntransformed = [{'name': r['name'], 'grade': 'A' if r['score'] >= 80 else 'B'} for r in raw]",
        )
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Alice: A" in output
        assert "Bob: A" in output

    def test_etl_edit_extract(self, nb_runner):
        """ETL pipeline, edit extract step."""
        nb_runner.create_notebook(
            [
                "# Extract\ndata = [10, 20, 30]",
                "# Transform\nscaled = [x * 2 for x in data]",
                "# Load\nresult = sum(scaled)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(3)

        # Edit extract
        nb_runner.set_cell_source(1, "# Extract v2\ndata = [100, 200, 300, 400]")
        nb_runner.run_all()
        assert "result = 2000" in nb_runner.get_output(3)


# Recursive function interaction tests.
#
# Tests editing recursive function definitions, base cases,
# and recursive steps, verifying correct recomputation.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRecursiveFunctionEdits:
    """Editing recursive function bodies."""

    def test_edit_base_case(self, nb_runner):
        """Edit the base case of a recursive function."""
        nb_runner.create_notebook(
            [
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "result = factorial(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

        # Change base case to return 2
        nb_runner.set_cell_source(
            1,
            "def factorial(n):\n    if n <= 1:\n        return 2\n    return n * factorial(n - 1)",
        )
        nb_runner.run_all()
        assert "result = 240" in nb_runner.get_output(2)

    def test_edit_recursive_step(self, nb_runner):
        """Edit the recursive step."""
        nb_runner.create_notebook(
            [
                "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
                "result = fib(7)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 13" in nb_runner.get_output(2)

        # Change to tribonacci
        nb_runner.set_cell_source(
            1,
            "def fib(n):\n    if n <= 1:\n        return n\n    if n == 2:\n        return 1\n    return fib(n-1) + fib(n-2) + fib(n-3)",
        )
        nb_runner.run_all()
        # tribonacci(7) = 24
        assert "result = 24" in nb_runner.get_output(2)

    def test_add_memoization(self, nb_runner):
        """Add memoization to a recursive function."""
        nb_runner.create_notebook(
            [
                "def slow_sum(n):\n    if n <= 0:\n        return 0\n    return n + slow_sum(n - 1)",
                "result = slow_sum(100)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5050" in nb_runner.get_output(2)

        # Add caching
        nb_runner.set_cell_source(
            1,
            "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef slow_sum(n):\n    if n <= 0:\n        return 0\n    return n + slow_sum(n - 1)",
        )
        nb_runner.run_all()
        assert "result = 5050" in nb_runner.get_output(2)

    def test_recursive_with_helper(self, nb_runner):
        """Edit a recursive function that calls a helper."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "def recurse(n):\n    if n <= 0:\n        return 0\n    return double(n) + recurse(n - 1)",
                "result = recurse(4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # double(4)+double(3)+double(2)+double(1) = 8+6+4+2 = 20
        assert "result = 20" in nb_runner.get_output(3)

        # Edit helper
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 3")
        nb_runner.run_all()
        # 12+9+6+3 = 30
        assert "result = 30" in nb_runner.get_output(3)


# Exception hierarchy and error handling edits.
#
# Tests custom exception classes and try/except flow with edits.
@pytest.mark.timeout(90)
class TestExceptionHierarchyEdit:
    """Custom exception and error handling patterns."""

    def test_custom_exception_edit(self, nb_runner):
        """Edit custom exception message format."""
        nb_runner.create_notebook(
            [
                "class AppError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        self.msg = msg\n    def __str__(self):\n        return f'Error {self.code}: {self.msg}'",
                "try:\n    raise AppError(404, 'not found')\nexcept AppError as e:\n    result = str(e)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = Error 404: not found" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class AppError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        self.msg = msg\n    def __str__(self):\n        return f'[{self.code}] {self.msg}'",
        )
        nb_runner.run_all()
        assert "result = [404] not found" in nb_runner.get_output(2)

    def test_exception_handler_edit(self, nb_runner):
        """Edit the exception handler logic."""
        nb_runner.create_notebook(
            [
                "def safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return -1",
                "result = safe_div(10, 0)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return 0",
        )
        nb_runner.run_all()
        assert "result = 0" in nb_runner.get_output(2)

    def test_multiple_except_edit(self, nb_runner):
        """Edit multi-handler except block."""
        nb_runner.create_notebook(
            [
                "def parse_value(s):\n    try:\n        return int(s)\n    except ValueError:\n        return 'not_int'\n    except TypeError:\n        return 'bad_type'",
                "r1 = parse_value('abc')\nr2 = parse_value(None)\nprint(f'r1={r1} r2={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=not_int r2=bad_type" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def parse_value(s):\n    try:\n        return int(s)\n    except ValueError:\n        return -1\n    except TypeError:\n        return -2",
        )
        nb_runner.run_all()
        assert "r1=-1 r2=-2" in nb_runner.get_output(2)


# math module chain operations with caching and edit propagation.
# Tests math.sqrt, math.pow, math.log chains and invalidation on edit.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestMathChainEdit:
    """Test math module chain operations caching."""

    def test_math_sqrt_chain(self, nb_runner):
        """Chain math.sqrt operations, verify caching."""
        nb_runner.create_notebook(
            [
                "import math",
                "x = 256",
                "y = math.sqrt(x)\nz = math.sqrt(y)",
                "print(f'z={z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "z=4.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "z=4.0" in out2

    def test_math_pow_log_edit(self, nb_runner):
        """Edit base value, propagate through pow/log chain."""
        nb_runner.create_notebook(
            [
                "import math",
                "base = 2",
                "powered = math.pow(base, 10)\nlog_val = math.log2(powered)",
                "result = int(log_val)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "base = 3")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        # log2(3^10) = 10 * log2(3) ≈ 15.849
        val = int(float(out2.split("result=")[1].strip()))
        assert val == 15  # int truncation of 15.849

    def test_math_trig_chain(self, nb_runner):
        """Trigonometric chain with pi."""
        nb_runner.create_notebook(
            [
                "import math",
                "angle = math.pi / 4",
                "s = math.sin(angle)\nc = math.cos(angle)\nidentity = round(s**2 + c**2, 10)",
                "print(f'identity={identity}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "identity=1.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "identity=1.0" in out2


# Zip and enumerate patterns with edits.
#
# Tests zip, enumerate, and parallel iteration with data edits.
@pytest.mark.timeout(90)
class TestZipEnumEdits:
    """Zip/enumerate edit patterns."""

    def test_zip_edit_one_list(self, nb_runner):
        """Edit one of two zipped lists."""
        nb_runner.create_notebook(
            [
                "names = ['Alice', 'Bob', 'Charlie']",
                "scores = [90, 85, 78]",
                "pairs = list(zip(names, scores))\nprint(f'pairs = {pairs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('Alice', 90)" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "scores = [100, 95, 88]")
        nb_runner.run_all()
        assert "('Alice', 100)" in nb_runner.get_output(3)
        assert "('Charlie', 88)" in nb_runner.get_output(3)

    def test_enumerate_with_edit(self, nb_runner):
        """Edit list, enumerate indexes correctly reflect."""
        nb_runner.create_notebook(
            [
                "items = ['apple', 'banana', 'cherry']",
                "indexed = list(enumerate(items, start=1))\nprint(f'indexed = {indexed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(1, 'apple')" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "items = ['x', 'y']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(1, 'x')" in out
        assert "(2, 'y')" in out

    def test_zip_longest_edit(self, nb_runner):
        """Edit data in zip_longest scenario."""
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\na = [1, 2, 3]\nb = ['x', 'y']",
                "result = list(zip_longest(a, b, fillvalue='?'))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(3, '?')" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "from itertools import zip_longest\na = [1]\nb = ['x', 'y', 'z']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(1, 'x')" in out
        assert "('?', 'z')" in out


# Async/await patterns with edits.
#
# Tests asyncio-based patterns in notebook cells.
@pytest.mark.timeout(90)
class TestAsyncPatterns:
    """Async/await edit propagation."""

    def test_async_function_edit(self, nb_runner):
        """Edit async function, await result changes."""
        nb_runner.create_notebook(
            [
                "import asyncio\nasync def compute(x):\n    return x * 2",
                "result = await compute(21)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "import asyncio\nasync def compute(x):\n    return x ** 2",
        )
        nb_runner.run_all()
        assert "result = 441" in nb_runner.get_output(2)

    def test_async_gather_edit(self, nb_runner):
        """Edit async tasks gathered together."""
        nb_runner.create_notebook(
            [
                "import asyncio\nasync def task(n):\n    return n + 1",
                "result = list(await asyncio.gather(task(1), task(2), task(3)))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "import asyncio\nasync def task(n):\n    return n * 10",
        )
        nb_runner.run_all()
        assert "result = [10, 20, 30]" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestRegexPatternEdit:
    """re (regex) pattern matching with cell edits."""

    def test_regex_findall(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ntext = 'Call 555-1234 or 555-5678 for info'",
                "pattern = r'\\d{3}-\\d{4}'\nnumbers = re.findall(pattern, text)\nprint(f'numbers={numbers}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "numbers=['555-1234', '555-5678']" in nb_runner.get_output(2)

    def test_regex_edit_text(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ntext = 'apple 3 banana 7 cherry 12'",
                "nums = [int(x) for x in re.findall(r'\\d+', text)]\ntotal = sum(nums)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=22" in nb_runner.get_output(2)
        # Edit text
        nb_runner.set_cell_source(1, "import re\ntext = 'x 100 y 200 z 300'")
        nb_runner.run_all()
        assert "total=600" in nb_runner.get_output(2)


# Itertools patterns with edits.
#
# Tests itertools functions with data/function edits.
@pytest.mark.timeout(90)
class TestItertoolsEdits:
    """Itertools operation edit patterns."""

    def test_groupby_edit(self, nb_runner):
        """Edit data before groupby."""
        nb_runner.create_notebook(
            [
                "from itertools import groupby\ndata = [('a', 1), ('a', 2), ('b', 3), ('b', 4)]",
                "groups = {k: list(v) for k, v in groupby(data, key=lambda x: x[0])}\nresult = {k: len(v) for k, v in groups.items()}\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 2" in out
        assert "'b': 2" in out

        nb_runner.set_cell_source(
            1,
            "from itertools import groupby\ndata = [('x', 1), ('x', 2), ('x', 3), ('y', 4)]",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "'x': 3" in out2
        assert "'y': 1" in out2
