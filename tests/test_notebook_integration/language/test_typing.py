"""typing, protocols and type conversion across cells."""

import textwrap

import pytest


# Type system and typing patterns — type hints, TypeVar, Generic,
# Protocol, Union, Optional, Literal across cells.
@pytest.mark.integration
@pytest.mark.stress
class TestTypeHintPatterns:
    """Test caching with type annotations in code."""

    def test_optional_annotation(self, nb_runner):
        """Optional type annotation."""
        nb_runner.create_notebook(
            [
                "from typing import Optional",
                textwrap.dedent("""\
                def find(items: list, key: str) -> Optional[int]:
                    for i, item in enumerate(items):
                        if item == key:
                            return i
                    return None
            """),
                textwrap.dedent("""\
                idx = find(['a', 'b', 'c'], 'b')
                miss = find(['a', 'b', 'c'], 'z')
                print(f"idx={idx} miss={miss}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "idx=1 miss=None" in nb_runner.get_output(3)


# Error handling patterns, type annotations, abstract classes,
# metaclass interactions, and exception flow caching.
#
# Tests how cash handles try/except, custom exceptions, type-annotated code,
# abstract base classes, metaclass-driven class creation, and exception
# propagation across cells.
@pytest.mark.integration
@pytest.mark.stress
class TestTypeAnnotations:
    """Test that type-annotated code caches correctly."""

    def test_typed_function(self, nb_runner):
        """Function with type annotations."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def add(x: int, y: int) -> int:
                    return x + y

                result: int = add(3, 4)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(1)


# Type annotation and typed data patterns.
#
# Tests typed function signatures with edits, verifying runtime behavior.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTypeAnnotationPatterns:
    """Typed function and data patterns with edits."""

    def test_typed_function_edit(self, nb_runner):
        """Edit typed function body, return type stays consistent."""
        nb_runner.create_notebook(
            [
                "def process(items: list[int]) -> int:\n    return sum(items)",
                "data: list[int] = [10, 20, 30]\nresult: int = process(data)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def process(items: list[int]) -> int:\n    return max(items) - min(items)",
        )
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

    def test_typed_dict_pattern(self, nb_runner):
        """Edit TypedDict-like dict, downstream reflects."""
        nb_runner.create_notebook(
            [
                "record: dict[str, int | str] = {'name': 'Alice', 'age': 30, 'score': 95}",
                "summary = f\"{record['name']}: age={record['age']}, score={record['score']}\"\nprint(f'summary = {summary}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary = Alice: age=30, score=95" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "record: dict[str, int | str] = {'name': 'Bob', 'age': 25, 'score': 88}",
        )
        nb_runner.run_all()
        assert "summary = Bob: age=25, score=88" in nb_runner.get_output(2)

    def test_optional_type_edit(self, nb_runner):
        """Edit function with Optional param."""
        nb_runner.create_notebook(
            [
                "def greet(name: str, title: str | None = None) -> str:\n    if title:\n        return f'{title} {name}'\n    return name",
                "msg = greet('Alice', 'Dr')\nprint(f'msg = {msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = Dr Alice" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def greet(name: str, title: str | None = None) -> str:\n    if title:\n        return f'{title}. {name}'\n    return f'Dear {name}'",
        )
        nb_runner.run_all()
        assert "msg = Dr. Alice" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTypingPatterns:
    """typing module usage patterns and class annotations."""

    def test_typed_dict_usage(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import Dict, List, Tuple\ndef aggregate(records: List[Tuple[str, int]]) -> Dict[str, int]:\n    result: Dict[str, int] = {}\n    for name, val in records:\n        result[name] = result.get(name, 0) + val\n    return result",
                "data = [('a', 10), ('b', 20), ('a', 5)]\nagg = aggregate(data)\nprint(f'agg={dict(sorted(agg.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "agg={'a': 15, 'b': 20}" in nb_runner.get_output(2)

    def test_typed_function_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import Optional\ndef safe_div(a: float, b: float) -> Optional[float]:\n    if b == 0:\n        return None\n    return a / b",
                "r1 = safe_div(10, 3)\nr2 = safe_div(5, 0)\nprint(f'r1={round(r1, 2) if r1 else r1} r2={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=3.33 r2=None" in nb_runner.get_output(2)
        # Edit function
        nb_runner.set_cell_source(
            1,
            "from typing import Optional\ndef safe_div(a: float, b: float) -> Optional[float]:\n    if b == 0:\n        return -1.0\n    return a / b",
        )
        nb_runner.run_all()
        assert "r1=3.33 r2=-1.0" in nb_runner.get_output(2)

    def test_generic_container(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import List\ndef flatten(nested: List[List[int]]) -> List[int]:\n    return [item for sub in nested for item in sub]",
                "data = [[1, 2], [3, 4], [5]]\nresult = flatten(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)


# Interaction test: typing module with TypeVar, Generic, Protocol.
# Tests type annotation patterns used in data science code,
# cross-cell generic class usage, and cache invalidation.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTypingGenericProtocol:
    """Test typing constructs across cells."""

    def test_typing_generic(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: generic stack
                "from typing import Generic, TypeVar, List\nT = TypeVar('T')\nclass Stack(Generic[T]):\n    def __init__(self):\n        self._items: List[T] = []\n    def push(self, item: T) -> None:\n        self._items.append(item)\n    def pop(self) -> T:\n        return self._items.pop()\n    def size(self) -> int:\n        return len(self._items)\n\ns = Stack()\ns.push(10)\ns.push(20)\ns.push(30)\nprint(f'size={s.size()}')",
                # Cell 2: use stack
                "top = s.pop()\nprint(f'top={top}')\nprint(f'size_after={s.size()}')",
                # Cell 3: string stack
                "ss = Stack()\nfor w in ['hello', 'world']:\n    ss.push(w)\nprint(f'str_size={ss.size()}')\nprint(f'str_top={ss.pop()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "size=3" in out1
        out2 = nb_runner.get_output(2)
        assert "top=30" in out2
        assert "size_after=2" in out2
        out3 = nb_runner.get_output(3)
        assert "str_size=2" in out3
        assert "str_top=world" in out3

    def test_typing_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: float\n    y: float\np = Point(3.0, 4.0)\nprint(f'point={p}')",
                "dist = (p.x ** 2 + p.y ** 2) ** 0.5\nprint(f'dist={dist}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "point=Point(x=3.0, y=4.0)" in nb_runner.get_output(1)
        assert "dist=5.0" in nb_runner.get_output(2)

        # Edit point
        nb_runner.set_cell_source(
            1,
            "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: float\n    y: float\np = Point(5.0, 12.0)\nprint(f'point={p}')",
        )
        nb_runner.run_cells([1, 2])
        assert "point=Point(x=5.0, y=12.0)" in nb_runner.get_output(1)
        assert "dist=13.0" in nb_runner.get_output(2)

    def test_typing_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import Dict, Tuple\ndef make_pair(k: str, v: int) -> Tuple[str, int]:\n    return (k, v)\npair = make_pair('age', 25)\nprint(f'pair={pair}')",
                "key_val = f'{pair[0]}={pair[1]}'\nprint(f'kv={key_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pair=('age', 25)" in nb_runner.get_output(1)
        assert "kv=age=25" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "kv=age=25" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestCallableTypePatterns:
    """Test Callable type patterns."""

    def test_callable_change_propagation(self, nb_runner):
        """Change callable parameter → output updates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def transform(fn, data):
                    return [fn(x) for x in data]
            """),
                "op = lambda x: x + 1",
                textwrap.dedent("""\
                result = transform(op, [10, 20, 30])
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[11, 21, 31]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "op = lambda x: x * 10")
        nb_runner.run_all()
        assert "[100, 200, 300]" in nb_runner.get_output(3)


# Typing module & Protocol patterns — cash caching with type annotations.
@pytest.mark.stress
class TestProtocolPatterns:
    """Test Protocol-based structural typing."""

    def test_typed_change_propagation(self, nb_runner):
        """Type-annotated variables propagate on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from typing import Dict, List

                scores: Dict[str, List[int]] = {
                    'math': [90, 85, 92],
                    'science': [88, 91, 87],
                }
            """),
                textwrap.dedent("""\
                averages = {k: sum(v) / len(v) for k, v in scores.items()}
                print(f"averages={averages}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "math" in out
        assert "89.0" in out

        # Add a subject
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from typing import Dict, List

            scores: Dict[str, List[int]] = {
                'math': [90, 85, 92],
                'science': [88, 91, 87],
                'english': [95, 90, 88],
            }
        """),
        )
        nb_runner.run_cells([1, 2])
        out2 = nb_runner.get_output(2)
        assert "english" in out2
        assert "91.0" in out2


# Protocol/interface interaction tests.
#
# Tests editing cells with abstract base class and protocol
# patterns and verifying downstream propagation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestProtocolPatternEdits:
    """Editing protocol/interface patterns."""

    def test_edit_abc_implementation(self, nb_runner):
        """Edit an ABC-derived class implementation."""
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass",
                "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14 * self.r ** 2",
                "c = Circle(5)\nprint(f'area = {c.area()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.5" in nb_runner.get_output(3)

        # Edit Circle implementation
        nb_runner.set_cell_source(
            2,
            "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14159 * self.r ** 2",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "78.5" in out or "78.53" in out

    def test_edit_strategy_pattern(self, nb_runner):
        """Edit strategy function selection."""
        nb_runner.create_notebook(
            [
                "def add(a, b):\n    return a + b\ndef multiply(a, b):\n    return a * b",
                "strategy = add\nresult = strategy(3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)

        # Switch strategy
        nb_runner.set_cell_source(2, "strategy = multiply\nresult = strategy(3, 4)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    def test_edit_interface_method(self, nb_runner):
        """Edit a class that implements a protocol."""
        nb_runner.create_notebook(
            [
                "class Formatter:\n    def format(self, text):\n        return text.upper()",
                "f = Formatter()\nprint(f.format('hello world'))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HELLO WORLD" in nb_runner.get_output(2)

        # Change formatting
        nb_runner.set_cell_source(1, "class Formatter:\n    def format(self, text):\n        return text.title()")
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(2)


# Protocol / structural subtyping interaction tests.
# Tests that editing classes implementing protocols properly invalidates
# downstream cells that use protocol-based operations.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestProtocolInteraction:
    """Test protocol/structural subtyping patterns with cache invalidation."""

    def test_duck_typing_edit(self, nb_runner):
        """Editing a class used via duck typing should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Dog:\n"
                    "    def speak(self):\n"
                    "        return 'Woof'\n"
                    "class Cat:\n"
                    "    def speak(self):\n"
                    "        return 'Meow'"
                ),
                "animals = [Dog(), Cat()]",
                "sounds = [a.speak() for a in animals]",
                "result = ', '.join(sounds)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=Woof, Meow" in out

        # Edit class definitions
        nb_runner.set_cell_source(
            1,
            (
                "class Dog:\n"
                "    def speak(self):\n"
                "        return 'BARK'\n"
                "class Cat:\n"
                "    def speak(self):\n"
                "        return 'HISS'"
            ),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=BARK, HISS" in out

    def test_callable_protocol_edit(self, nb_runner):
        """Editing callable objects used as strategies should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Adder:\n"
                    "    def __init__(self, n):\n"
                    "        self.n = n\n"
                    "    def __call__(self, x):\n"
                    "        return x + self.n"
                ),
                "op = Adder(10)",
                "result = op(5)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=15" in out

        nb_runner.set_cell_source(2, "op = Adder(100)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=105" in out

    def test_iterable_protocol_edit(self, nb_runner):
        """Editing a custom iterable class should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Range2:\n"
                    "    def __init__(self, start, stop):\n"
                    "        self.start = start\n"
                    "        self.stop = stop\n"
                    "    def __iter__(self):\n"
                    "        current = self.start\n"
                    "        while current < self.stop:\n"
                    "            yield current\n"
                    "            current += 2"
                ),
                "r = Range2(0, 10)",
                "vals = list(r)",
                "result = ','.join(str(v) for v in vals)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=0,2,4,6,8" in out

        nb_runner.set_cell_source(2, "r = Range2(1, 12)")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=1,3,5,7,9,11" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTypingProtocolRuntime:
    """typing protocol and runtime checkable."""

    def test_protocol_duck_typing(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import Protocol, runtime_checkable",
                "@runtime_checkable\nclass Drawable(Protocol):\n    def draw(self) -> str: ...\nclass Circle:\n    def draw(self) -> str: return 'O'\nclass Square:\n    def draw(self) -> str: return '[]'\nc = Circle()\ns = Square()\nprint(f'c_drawable={isinstance(c, Drawable)} s_drawable={isinstance(s, Drawable)}')\nprint(f'c={c.draw()} s={s.draw()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c_drawable=True" in out
        assert "s_drawable=True" in out
        assert "c=O" in out
        assert "s=[]" in out

    def test_non_conforming(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import Protocol, runtime_checkable",
                "@runtime_checkable\nclass Serializable(Protocol):\n    def to_json(self) -> str: ...\nclass Good:\n    def to_json(self) -> str: return '{}'\nclass Bad:\n    pass\nprint(f'good={isinstance(Good(), Serializable)} bad={isinstance(Bad(), Serializable)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "good=True" in out
        assert "bad=False" in out

    def test_protocol_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import Protocol, runtime_checkable",
                "@runtime_checkable\nclass HasLen(Protocol):\n    def __len__(self) -> int: ...\nprint(f'list={isinstance([], HasLen)} int={isinstance(5, HasLen)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "list=True" in out
        assert "int=False" in out
        nb_runner.set_cell_source(
            2,
            "@runtime_checkable\nclass HasLen(Protocol):\n    def __len__(self) -> int: ...\nprint(f'str={isinstance(\"hi\", HasLen)} dict={isinstance({}, HasLen)}')",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "str=True" in out2
        assert "dict=True" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTypeCheckingInstanceSubclass:
    """type checking with isinstance and issubclass."""

    def test_isinstance_multi(self, nb_runner):
        nb_runner.create_notebook(
            [
                "vals = [42, 'hello', 3.14, True, [1, 2]]",
                "types = [(v, type(v).__name__, isinstance(v, (int, float))) for v in vals]\nresults = [(t[1], t[2]) for t in types]\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('int', True)" in out
        assert "('str', False)" in out

    def test_issubclass(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Base: pass\nclass Mid(Base): pass\nclass Leaf(Mid): pass",
                "r1 = issubclass(Leaf, Base)\nr2 = issubclass(Leaf, Mid)\nr3 = issubclass(Base, Leaf)\nprint(f'r1={r1} r2={r2} r3={r3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=True" in nb_runner.get_output(2)
        assert "r2=True" in nb_runner.get_output(2)
        assert "r3=False" in nb_runner.get_output(2)

    def test_type_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "val = 42",
                "t = type(val).__name__\nis_num = isinstance(val, (int, float))\nprint(f'type={t} is_num={is_num}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type=int" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "val = 'hello'")
        nb_runner.run_all()
        assert "type=str" in nb_runner.get_output(2)
        assert "is_num=False" in nb_runner.get_output(2)


# Type conversion / coercion chain interaction tests.
#
# Tests editing type conversions (int→str→float, etc.),
# serialization round-trips, and format changes.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestTypeConversionEdits:
    """Editing type conversion chains."""

    def test_edit_conversion_chain(self, nb_runner):
        """Edit a type conversion chain."""
        nb_runner.create_notebook(
            [
                "raw = '42.5'  # type conversion source",
                "value = int(float(raw))\nprint(f'value = {value}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "value = 42" in nb_runner.get_output(2)

        # Change chain to round instead of truncate
        nb_runner.set_cell_source(2, "value = round(float(raw))\nprint(f'value = {value}')")
        nb_runner.run_all()
        assert "value = 42" in nb_runner.get_output(2)

        # Change source value
        nb_runner.set_cell_source(1, "raw = '42.7'  # type conversion source v2")
        nb_runner.run_all()
        assert "value = 43" in nb_runner.get_output(2)

    def test_edit_format_conversion(self, nb_runner):
        """Edit between different format representations."""
        nb_runner.create_notebook(
            [
                "number = 255  # format source",
                "formatted = hex(number)\nprint(f'formatted = {formatted}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted = 0xff" in nb_runner.get_output(2)

        # Change to binary
        nb_runner.set_cell_source(2, "formatted = bin(number)\nprint(f'formatted = {formatted}')")
        nb_runner.run_all()
        assert "formatted = 0b11111111" in nb_runner.get_output(2)

        # Change to octal
        nb_runner.set_cell_source(2, "formatted = oct(number)\nprint(f'formatted = {formatted}')")
        nb_runner.run_all()
        assert "formatted = 0o377" in nb_runner.get_output(2)


# Type conversion chain interaction tests.
# Tests str→int, list→tuple→set, and dict→items→sorted conversion chains.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTypeConvChainInteraction:
    """Test type conversion chains with cache invalidation."""

    def test_str_to_int_chain_edit(self, nb_runner):
        """Editing string input should propagate through int conversion."""
        nb_runner.create_notebook(
            [
                "raw = '42'",
                "val = int(raw)",
                "doubled = val * 2",
                "print(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "doubled=84" in out

        nb_runner.set_cell_source(1, "raw = '100'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "doubled=200" in out

    def test_list_to_tuple_to_set_edit(self, nb_runner):
        """Editing list should propagate through tuple and set conversions."""
        nb_runner.create_notebook(
            [
                "data = [3, 1, 4, 1, 5, 9, 2, 6]",
                "as_tuple = tuple(sorted(data))",
                "as_set = set(data)",
                "info = f'tuple_len={len(as_tuple)},set_len={len(as_set)}'",
                "print(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "tuple_len=8" in out
        assert "set_len=7" in out

        nb_runner.set_cell_source(1, "data = [1, 1, 1, 2, 2]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "tuple_len=5" in out
        assert "set_len=2" in out

    def test_dict_items_sorted_edit(self, nb_runner):
        """Editing dict should propagate through items/sorted chain."""
        nb_runner.create_notebook(
            [
                "mapping = {'b': 2, 'a': 1, 'c': 3}",
                "items = list(mapping.items())",
                "sorted_items = sorted(items)",
                "keys = [k for k, v in sorted_items]",
                "print(f'keys={keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "keys=['a', 'b', 'c']" in out

        nb_runner.set_cell_source(1, "mapping = {'z': 26, 'x': 24, 'y': 25}")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "keys=['x', 'y', 'z']" in out
