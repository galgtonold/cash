"""Batch 53: Iterator & custom container patterns — __iter__, __getitem__, __contains__."""
import textwrap
import pytest


@pytest.mark.stress
class TestCustomIterators:
    """Test custom iterator classes."""

    def test_range_iterator(self, nb_runner):
        """Custom range-like iterator across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class FibRange:
                    def __init__(self, n):
                        self.n = n
                    def __iter__(self):
                        a, b, count = 0, 1, 0
                        while count < self.n:
                            yield a
                            a, b = b, a + b
                            count += 1

                fib = FibRange(8)
            """),
            textwrap.dedent("""\
                values = list(fib)
                print(f"values={values}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "values=[0, 1, 1, 2, 3, 5, 8, 13]" in nb_runner.get_output(2)

    def test_infinite_iterator_sliced(self, nb_runner):
        """Infinite iterator with islice."""
        nb_runner.create_notebook([
            "from itertools import islice",
            textwrap.dedent("""\
                def powers_of_two():
                    n = 1
                    while True:
                        yield n
                        n *= 2

                gen = powers_of_two()
                first_10 = list(islice(gen, 10))
                print(f"first_10={first_10}")
            """),
            textwrap.dedent("""\
                total = sum(first_10)
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first_10=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]" in nb_runner.get_output(2)
        assert "total=1023" in nb_runner.get_output(3)

    def test_chained_itertools(self, nb_runner):
        """Complex itertools chains cached across cells."""
        nb_runner.create_notebook([
            "from itertools import chain, repeat, cycle, islice",
            textwrap.dedent("""\
                pattern = list(islice(cycle([1, 2, 3]), 9))
                repeated = list(chain(repeat('a', 3), repeat('b', 2)))
                print(f"pattern={pattern} repeated={repeated}")
            """),
            textwrap.dedent("""\
                combined = list(zip(pattern, repeated * 2))[:5]
                print(f"combined={combined}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pattern=[1, 2, 3, 1, 2, 3, 1, 2, 3]" in nb_runner.get_output(2)


@pytest.mark.stress
class TestCustomContainers:
    """Test custom container classes."""

    def test_ordered_set(self, nb_runner):
        """Custom OrderedSet-like container."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class OrderedSet:
                    def __init__(self, items=None):
                        self._items = []
                        self._set = set()
                        if items:
                            for item in items:
                                self.add(item)

                    def add(self, item):
                        if item not in self._set:
                            self._items.append(item)
                            self._set.add(item)

                    def __contains__(self, item):
                        return item in self._set

                    def __len__(self):
                        return len(self._items)

                    def __iter__(self):
                        return iter(self._items)

                    def __repr__(self):
                        return f"OrderedSet({self._items})"
            """),
            textwrap.dedent("""\
                os = OrderedSet([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
                print(f"os={os} len={len(os)}")
                print(f"has_5={5 in os} has_7={7 in os}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "OrderedSet([3, 1, 4, 5, 9, 2, 6])" in out
        assert "len=7" in out
        assert "has_5=True has_7=False" in out

    def test_matrix_container(self, nb_runner):
        """Custom matrix with __getitem__ and __setitem__."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Matrix(3x3)" in nb_runner.get_output(2)
        assert "diag=[1, 1, 1]" in nb_runner.get_output(2)
        assert "off=0" in nb_runner.get_output(2)

    def test_default_dict_like(self, nb_runner):
        """Custom defaultdict-like with factory across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('the', 3)" in out
        assert "('cat', 2)" in out

    def test_stack_implementation(self, nb_runner):
        """Stack data structure across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Stack:
                    def __init__(self):
                        self._items = []
                    def push(self, item):
                        self._items.append(item)
                        return self
                    def pop(self):
                        return self._items.pop()
                    def peek(self):
                        return self._items[-1] if self._items else None
                    def __len__(self):
                        return len(self._items)
                    def __repr__(self):
                        return f"Stack({self._items})"

                s = Stack()
                s.push(10).push(20).push(30)
                print(f"stack={s} len={len(s)} peek={s.peek()}")
            """),
            textwrap.dedent("""\
                popped = s.pop()
                print(f"popped={popped} remaining={s}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Stack([10, 20, 30])" in nb_runner.get_output(1)
        assert "peek=30" in nb_runner.get_output(1)
        assert "popped=30" in nb_runner.get_output(2)
