"""
Batch 33: Type system and typing patterns — type hints, TypeVar, Generic,
Protocol, Union, Optional, Literal across cells.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestTypeHintPatterns:
    """Test caching with type annotations in code."""

    def test_annotated_function(self, nb_runner):
        """Function with type annotations cached correctly."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def add(a: int, b: int) -> int:
                    return a + b
            """),
            textwrap.dedent("""\
                result: int = add(3, 4)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(2)

    def test_optional_annotation(self, nb_runner):
        """Optional type annotation."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "idx=1 miss=None" in nb_runner.get_output(3)

    def test_generic_class(self, nb_runner):
        """Generic class with TypeVar."""
        nb_runner.create_notebook([
            "from typing import TypeVar, Generic, List",
            textwrap.dedent("""\
                T = TypeVar('T')
                class Stack(Generic[T]):
                    def __init__(self):
                        self._items: List[T] = []
                    def push(self, item: T) -> None:
                        self._items.append(item)
                    def pop(self) -> T:
                        return self._items.pop()
                    def __len__(self) -> int:
                        return len(self._items)
            """),
            textwrap.dedent("""\
                s: Stack[int] = Stack()
                s.push(1)
                s.push(2)
                s.push(3)
                top = s.pop()
                print(f"top={top} len={len(s)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top=3 len=2" in nb_runner.get_output(3)

    def test_union_type(self, nb_runner):
        """Union types across cells."""
        nb_runner.create_notebook([
            "from typing import Union",
            textwrap.dedent("""\
                def stringify(val: Union[int, float, str]) -> str:
                    if isinstance(val, float):
                        return f"{val:.2f}"
                    return str(val)
            """),
            textwrap.dedent("""\
                r1 = stringify(42)
                r2 = stringify(3.14)
                r3 = stringify("hello")
                print(f"{r1} {r2} {r3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42 3.14 hello" in nb_runner.get_output(3)


class TestTypedDictPattern:
    """Test TypedDict patterns."""

    def test_typed_dict(self, nb_runner):
        """TypedDict across cells."""
        nb_runner.create_notebook([
            "from typing import TypedDict",
            textwrap.dedent("""\
                class UserInfo(TypedDict):
                    name: str
                    age: int
                    active: bool
            """),
            textwrap.dedent("""\
                user: UserInfo = {'name': 'Alice', 'age': 30, 'active': True}
                print(f"{user['name']} age={user['age']}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice age=30" in nb_runner.get_output(3)


class TestTypeCheckingPatterns:
    """Test patterns commonly used with mypy/pyright."""

    def test_type_guard_pattern(self, nb_runner):
        """isinstance-based type narrowing."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def process(val):
                    if isinstance(val, int):
                        return val * 2
                    elif isinstance(val, str):
                        return val.upper()
                    elif isinstance(val, list):
                        return len(val)
                    return None
            """),
            textwrap.dedent("""\
                r1 = process(5)
                r2 = process("hello")
                r3 = process([1, 2, 3])
                print(f"{r1} {r2} {r3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10 HELLO 3" in nb_runner.get_output(2)

    def test_abstract_base_with_annotations(self, nb_runner):
        """ABC with type annotations across cells."""
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod",
            textwrap.dedent("""\
                class Shape(ABC):
                    @abstractmethod
                    def area(self) -> float:
                        ...
                    @abstractmethod
                    def perimeter(self) -> float:
                        ...
            """),
            textwrap.dedent("""\
                class Square(Shape):
                    def __init__(self, side: float):
                        self.side = side
                    def area(self) -> float:
                        return self.side ** 2
                    def perimeter(self) -> float:
                        return 4 * self.side
            """),
            textwrap.dedent("""\
                s: Shape = Square(5)
                print(f"area={s.area()} perim={s.perimeter()}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=25 perim=20" in nb_runner.get_output(4)


class TestCallableTypePatterns:
    """Test Callable type patterns."""

    def test_callable_parameter(self, nb_runner):
        """Function accepting Callable parameter."""
        nb_runner.create_notebook([
            "from typing import Callable, List",
            textwrap.dedent("""\
                def apply_to_all(fn: Callable[[int], int], items: List[int]) -> List[int]:
                    return [fn(x) for x in items]
            """),
            textwrap.dedent("""\
                doubled = apply_to_all(lambda x: x * 2, [1, 2, 3, 4])
                print(doubled)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8]" in nb_runner.get_output(3)

    def test_callable_change_propagation(self, nb_runner):
        """Change callable parameter → output updates."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def transform(fn, data):
                    return [fn(x) for x in data]
            """),
            "op = lambda x: x + 1",
            textwrap.dedent("""\
                result = transform(op, [10, 20, 30])
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[11, 21, 31]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "op = lambda x: x * 10")
        nb_runner.run_all()
        assert "[100, 200, 300]" in nb_runner.get_output(3)
