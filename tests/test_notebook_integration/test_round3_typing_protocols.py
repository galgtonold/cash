"""Batch 65: Typing module & Protocol patterns — cash caching with type annotations."""
import textwrap
import pytest


@pytest.mark.stress
class TestTypingBasics:
    """Test typing patterns across cells."""

    def test_typed_dict(self, nb_runner):
        """TypedDict usage across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from typing import TypedDict, List

                class Employee(TypedDict):
                    name: str
                    age: int
                    department: str

                employees: List[Employee] = [
                    {'name': 'Alice', 'age': 30, 'department': 'Eng'},
                    {'name': 'Bob', 'age': 25, 'department': 'Sales'},
                    {'name': 'Charlie', 'age': 35, 'department': 'Eng'},
                ]
                print(f"count={len(employees)}")
            """),
            textwrap.dedent("""\
                eng_team = [e for e in employees if e['department'] == 'Eng']
                avg_age = sum(e['age'] for e in eng_team) / len(eng_team)
                print(f"eng_count={len(eng_team)} avg_age={avg_age}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(1)
        assert "eng_count=2" in nb_runner.get_output(2)
        assert "avg_age=32.5" in nb_runner.get_output(2)

    def test_generic_class(self, nb_runner):
        """Generic class with type parameters."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from typing import Generic, TypeVar

                T = TypeVar('T')

                class Stack(Generic[T]):
                    def __init__(self):
                        self._items: list[T] = []
                    def push(self, item: T) -> None:
                        self._items.append(item)
                    def pop(self) -> T:
                        return self._items.pop()
                    def peek(self) -> T:
                        return self._items[-1]
                    def __len__(self):
                        return len(self._items)

                stack: Stack[int] = Stack()
                for i in [10, 20, 30]:
                    stack.push(i)
                print(f"len={len(stack)} top={stack.peek()}")
            """),
            textwrap.dedent("""\
                popped = stack.pop()
                print(f"popped={popped} remaining={len(stack)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3 top=30" in nb_runner.get_output(1)
        assert "popped=30 remaining=2" in nb_runner.get_output(2)


@pytest.mark.stress
class TestProtocolPatterns:
    """Test Protocol-based structural typing."""

    def test_protocol_duck_typing(self, nb_runner):
        """Protocol for structural typing across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from typing import Protocol, runtime_checkable

                @runtime_checkable
                class Drawable(Protocol):
                    def draw(self) -> str: ...

                class Circle:
                    def __init__(self, r):
                        self.r = r
                    def draw(self) -> str:
                        return f"Circle(r={self.r})"

                class Square:
                    def __init__(self, s):
                        self.s = s
                    def draw(self) -> str:
                        return f"Square(s={self.s})"

                shapes = [Circle(5), Square(3), Circle(10)]
            """),
            textwrap.dedent("""\
                drawings = [s.draw() for s in shapes if isinstance(s, Drawable)]
                print(f"drawings={drawings}")
                print(f"all_drawable={all(isinstance(s, Drawable) for s in shapes)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Circle(r=5)" in out
        assert "Square(s=3)" in out
        assert "all_drawable=True" in out

    def test_named_tuple_typed(self, nb_runner):
        """NamedTuple with types across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from typing import NamedTuple

                class Coordinate(NamedTuple):
                    x: float
                    y: float
                    label: str = ""

                points = [
                    Coordinate(1.0, 2.0, "A"),
                    Coordinate(3.0, 4.0, "B"),
                    Coordinate(5.0, 6.0, "C"),
                ]
            """),
            textwrap.dedent("""\
                import math
                distances = []
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    d = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
                    distances.append(round(d, 4))
                print(f"distances={distances}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "distances=[2.8284, 2.8284]" in nb_runner.get_output(2)

    def test_typed_change_propagation(self, nb_runner):
        """Type-annotated variables propagate on change."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "math" in out
        assert "89.0" in out

        # Add a subject
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            from typing import Dict, List

            scores: Dict[str, List[int]] = {
                'math': [90, 85, 92],
                'science': [88, 91, 87],
                'english': [95, 90, 88],
            }
        """))
        nb_runner.run_cells([1, 2])
        out2 = nb_runner.get_output(2)
        assert "english" in out2
        assert "91.0" in out2
