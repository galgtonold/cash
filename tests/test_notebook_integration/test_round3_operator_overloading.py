"""Batch 63: Operator overloading & dunder methods — cash caching with custom operators."""
import textwrap
import pytest


@pytest.mark.stress
class TestArithmeticDunders:
    """Test arithmetic operator overloading across cells."""

    def test_vector_arithmetic(self, nb_runner):
        """Custom vector with arithmetic operators."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Vec:
                    def __init__(self, x, y):
                        self.x, self.y = x, y
                    def __add__(self, other):
                        return Vec(self.x + other.x, self.y + other.y)
                    def __sub__(self, other):
                        return Vec(self.x - other.x, self.y - other.y)
                    def __mul__(self, scalar):
                        return Vec(self.x * scalar, self.y * scalar)
                    def __repr__(self):
                        return f"Vec({self.x}, {self.y})"
                    def __eq__(self, other):
                        return self.x == other.x and self.y == other.y

                a = Vec(1, 2)
                b = Vec(3, 4)
            """),
            textwrap.dedent("""\
                c = a + b
                d = b - a
                e = a * 3
                print(f"sum={c} diff={d} scaled={e}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=Vec(4, 6)" in nb_runner.get_output(2)
        assert "diff=Vec(2, 2)" in nb_runner.get_output(2)
        assert "scaled=Vec(3, 6)" in nb_runner.get_output(2)

    def test_matrix_dunders(self, nb_runner):
        """Matrix with __matmul__ (@) operator."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Matrix:
                    def __init__(self, data):
                        self.data = data
                        self.rows = len(data)
                        self.cols = len(data[0]) if data else 0
                    def __matmul__(self, other):
                        result = [[0] * other.cols for _ in range(self.rows)]
                        for i in range(self.rows):
                            for j in range(other.cols):
                                for k in range(self.cols):
                                    result[i][j] += self.data[i][k] * other.data[k][j]
                        return Matrix(result)
                    def __repr__(self):
                        return f"Matrix({self.data})"

                A = Matrix([[1, 2], [3, 4]])
                B = Matrix([[5, 6], [7, 8]])
            """),
            textwrap.dedent("""\
                C = A @ B
                print(f"result={C.data}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[[19, 22], [43, 50]]" in nb_runner.get_output(2)


@pytest.mark.stress
class TestComparisonDunders:
    """Test comparison operators across cells."""

    def test_comparable_class(self, nb_runner):
        """Class with full comparison operators."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from functools import total_ordering

                @total_ordering
                class Temperature:
                    def __init__(self, celsius):
                        self.celsius = celsius
                    def __eq__(self, other):
                        return self.celsius == other.celsius
                    def __lt__(self, other):
                        return self.celsius < other.celsius
                    def __repr__(self):
                        return f"{self.celsius}°C"

                temps = [Temperature(20), Temperature(35), Temperature(15), Temperature(28)]
            """),
            textwrap.dedent("""\
                sorted_temps = sorted(temps)
                print(f"sorted={sorted_temps}")
                print(f"max={max(temps)} min={min(temps)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sorted=[15°C, 20°C, 28°C, 35°C]" in out
        assert "max=35°C" in out
        assert "min=15°C" in out

    def test_hash_and_eq(self, nb_runner):
        """Custom __hash__ and __eq__ for set/dict usage."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Point:
                    def __init__(self, x, y):
                        self.x, self.y = x, y
                    def __hash__(self):
                        return hash((self.x, self.y))
                    def __eq__(self, other):
                        return (self.x, self.y) == (other.x, other.y)
                    def __repr__(self):
                        return f"P({self.x},{self.y})"

                points = {Point(1, 2), Point(3, 4), Point(1, 2)}
                print(f"unique={len(points)}")
            """),
            textwrap.dedent("""\
                lookup = {Point(1, 2): 'A', Point(3, 4): 'B'}
                val = lookup[Point(1, 2)]
                print(f"lookup={val}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=2" in nb_runner.get_output(1)
        assert "lookup=A" in nb_runner.get_output(2)


@pytest.mark.stress
class TestContainerDunders:
    """Test container protocol dunders."""

    def test_custom_sequence(self, nb_runner):
        """Custom sequence with __getitem__, __len__, __contains__."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class RangeSeq:
                    def __init__(self, start, stop, step=1):
                        self._data = list(range(start, stop, step))
                    def __getitem__(self, idx):
                        return self._data[idx]
                    def __len__(self):
                        return len(self._data)
                    def __contains__(self, item):
                        return item in self._data
                    def __iter__(self):
                        return iter(self._data)

                seq = RangeSeq(0, 20, 3)
                print(f"len={len(seq)} contains_9={9 in seq} first={seq[0]}")
            """),
            textwrap.dedent("""\
                as_list = list(seq)
                sliced = seq[1:4]
                print(f"list={as_list}")
                print(f"slice={sliced}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=7" in nb_runner.get_output(1)
        assert "contains_9=True" in nb_runner.get_output(1)
        assert "first=0" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "[0, 3, 6, 9, 12, 15, 18]" in out2

    def test_change_propagation_dunders(self, nb_runner):
        """Operator result propagation after change."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=150 USD" in nb_runner.get_output(2)

        # Change amount
        nb_runner.set_cell_source(1, textwrap.dedent("""\
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
        """))
        nb_runner.run_cells([1, 2])
        assert "total=275 USD" in nb_runner.get_output(2)
