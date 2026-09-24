"""The operator module, operator overloading and chained comparisons."""

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOperatorModule:
    """operator module and itemgetter/attrgetter patterns."""

    def test_itemgetter_sort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import itemgetter\ndata = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]",
                "by_score = sorted(data, key=itemgetter(1), reverse=True)\nnames = [x[0] for x in by_score]\nprint(f'names={names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['Bob', 'Alice', 'Charlie']" in nb_runner.get_output(2)

    def test_attrgetter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import attrgetter\nclass Student:\n    def __init__(self, name, gpa):\n        self.name = name\n        self.gpa = gpa\n    def __repr__(self):\n        return f'{self.name}:{self.gpa}'",
                "students = [Student('A', 3.5), Student('B', 3.9), Student('C', 3.2)]",
                "ranked = sorted(students, key=attrgetter('gpa'), reverse=True)\nprint(f'ranked={ranked}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ranked=[B:3.9, A:3.5, C:3.2]" in nb_runner.get_output(3)
        # Edit students
        nb_runner.set_cell_source(2, "students = [Student('X', 4.0), Student('Y', 2.8)]")
        nb_runner.run_all()
        assert "ranked=[X:4.0, Y:2.8]" in nb_runner.get_output(3)

    def test_methodcaller(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import methodcaller\nwords = ['hello', 'WORLD', 'Python']",
                "upper_words = list(map(methodcaller('upper'), words))\nprint(f'upper={upper_words}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "upper=['HELLO', 'WORLD', 'PYTHON']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOperatorGetters:
    """operator module itemgetter and attrgetter."""

    def test_itemgetter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import itemgetter\ndata = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]",
                "by_score = sorted(data, key=itemgetter(1), reverse=True)\ntop = by_score[0]\nprint(f'top={top}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top=('Bob', 92)" in nb_runner.get_output(2)

    def test_attrgetter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import attrgetter\nclass Student:\n    def __init__(self, name, gpa):\n        self.name = name\n        self.gpa = gpa\nstudents = [Student('X', 3.5), Student('Y', 3.9), Student('Z', 3.2)]",
                "best = max(students, key=attrgetter('gpa'))\nprint(f'best={best.name} gpa={best.gpa}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=Y" in nb_runner.get_output(2)
        assert "gpa=3.9" in nb_runner.get_output(2)

    def test_itemgetter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import itemgetter\nrecords = [{'name': 'a', 'val': 3}, {'name': 'b', 'val': 1}, {'name': 'c', 'val': 2}]",
                "ordered = sorted(records, key=itemgetter('val'))\nfirst = ordered[0]['name']\nprint(f'first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=b" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            1,
            "from operator import itemgetter\nrecords = [{'name': 'x', 'val': 5}, {'name': 'y', 'val': 2}, {'name': 'z', 'val': 8}]",
        )
        nb_runner.run_all()
        assert "first=y" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOperatorItemgetterAttrgetter:
    """operator itemgetter attrgetter sort patterns."""

    def test_itemgetter_sort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import itemgetter",
                "data = [('Alice', 90), ('Bob', 85), ('Carol', 95)]\nby_score = sorted(data, key=itemgetter(1))\nby_name = sorted(data, key=itemgetter(0))\nprint(f'by_score={[n for n,s in by_score]}')\nprint(f'by_name={[n for n,s in by_name]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_score=['Bob', 'Alice', 'Carol']" in out
        assert "by_name=['Alice', 'Bob', 'Carol']" in out

    def test_attrgetter_sort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import attrgetter",
                "class Student:\n    def __init__(self, name, gpa): self.name, self.gpa = name, gpa\nstudents = [Student('A', 3.5), Student('B', 3.9), Student('C', 3.2)]\nby_gpa = sorted(students, key=attrgetter('gpa'), reverse=True)\nnames = [s.name for s in by_gpa]\nprint(f'names={names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['B', 'A', 'C']" in nb_runner.get_output(2)

    def test_itemgetter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import itemgetter",
                "data = [(1, 'a'), (3, 'c'), (2, 'b')]\nresult = sorted(data, key=itemgetter(0))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'a'), (2, 'b'), (3, 'c')]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "data = [(1, 'z'), (3, 'a'), (2, 'm')]\nresult = sorted(data, key=itemgetter(1))\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=[(3, 'a'), (2, 'm'), (1, 'z')]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOperatorMethodcaller:
    """Test operator.methodcaller across cells."""

    def test_methodcaller_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: methodcaller for string methods
                "from operator import methodcaller\nwords = ['hello', 'WORLD', 'Python']\nupper = list(map(methodcaller('upper'), words))\nlower = list(map(methodcaller('lower'), words))\nprint(f'upper={upper}')\nprint(f'lower={lower}')",
                # Cell 2: methodcaller with args
                "texts = ['hello world', 'foo bar baz', 'one two']\nsplit2 = list(map(methodcaller('split', ' ', 1), texts))\nprint(f'split2={split2}')",
                # Cell 3: sorting with methodcaller
                "items = ['banana', 'apple', 'cherry']\nsorted_by_len = sorted(items, key=methodcaller('__len__'))\nprint(f'by_len={sorted_by_len}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "upper=['HELLO', 'WORLD', 'PYTHON']" in out1
        assert "lower=['hello', 'world', 'python']" in out1
        out2 = nb_runner.get_output(2)
        assert "['hello', 'world']" in out2
        out3 = nb_runner.get_output(3)
        assert "by_len=['apple', 'banana', 'cherry']" in out3

    def test_methodcaller_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import methodcaller\ndata = ['  hello  ', '  world  ']\nstripped = list(map(methodcaller('strip'), data))\nprint(f'stripped={stripped}')",
                "joined = '-'.join(stripped)\nprint(f'joined={joined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "joined=hello-world" in nb_runner.get_output(2)

        # Edit to title instead of strip
        nb_runner.set_cell_source(
            1,
            "from operator import methodcaller\ndata = ['hello there', 'world here']\nstripped = list(map(methodcaller('title'), data))\nprint(f'stripped={stripped}')",
        )
        nb_runner.run_cells([1, 2])
        assert "joined=Hello There-World Here" in nb_runner.get_output(2)

    def test_methodcaller_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from operator import methodcaller\nnums_str = ['1', '2', '3']\npadded = list(map(methodcaller('zfill', 3), nums_str))\nprint(f'padded={padded}')",
                "joined = ','.join(padded)\nprint(f'joined={joined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "padded=['001', '002', '003']" in nb_runner.get_output(1)
        assert "joined=001,002,003" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "joined=001,002,003" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsReduceOperator:
    """functools reduce and operator module."""

    def test_reduce_sum(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce\nimport operator",
                "nums = [1, 2, 3, 4, 5]\nproduct = reduce(operator.mul, nums)\ntotal = reduce(operator.add, nums)\nprint(f'product={product} total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "product=120" in out
        assert "total=15" in out

    def test_reduce_nested(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce",
                "lists = [[1, 2], [3, 4], [5]]\nflat = reduce(lambda a, b: a + b, lists)\nprint(f'flat={flat} len={len(flat)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "flat=[1, 2, 3, 4, 5]" in out
        assert "len=5" in out

    def test_reduce_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce\nimport operator",
                "vals = [2, 3, 4]\nresult = reduce(operator.mul, vals)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=24" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "vals = [10, 20, 30]\nresult = reduce(operator.add, vals)\nprint(f'result={result}')"
        )
        nb_runner.run_all()
        assert "result=60" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOperatorOverloading:
    """custom __add__, __mul__ operator overloading."""

    def test_add_mul(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Vector:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __add__(self, other):\n        return Vector(self.x + other.x, self.y + other.y)\n    def __mul__(self, scalar):\n        return Vector(self.x * scalar, self.y * scalar)\n    def __repr__(self):\n        return f'V({self.x},{self.y})'",
                "v1 = Vector(1, 2)\nv2 = Vector(3, 4)\nv3 = v1 + v2\nv4 = v1 * 3\nprint(f'v3={v3} v4={v4}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v3=V(4,6)" in nb_runner.get_output(2)
        assert "v4=V(3,6)" in nb_runner.get_output(2)

    def test_overload_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Money:\n    def __init__(self, amount):\n        self.amount = amount\n    def __add__(self, other):\n        return Money(self.amount + other.amount)\n    def __repr__(self):\n        return f'${self.amount}'",
                "m1 = Money(10)\nm2 = Money(25)\ntotal = m1 + m2\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=$35" in nb_runner.get_output(2)
        # Edit class to add sub
        nb_runner.set_cell_source(
            1,
            "class Money:\n    def __init__(self, amount):\n        self.amount = amount\n    def __add__(self, other):\n        return Money(self.amount + other.amount)\n    def __sub__(self, other):\n        return Money(self.amount - other.amount)\n    def __repr__(self):\n        return f'${self.amount}'",
        )
        nb_runner.set_cell_source(2, "m1 = Money(50)\nm2 = Money(25)\ndiff = m1 - m2\nprint(f'diff={diff}')")
        nb_runner.run_all()
        assert "diff=$25" in nb_runner.get_output(2)

    def test_iadd(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Accumulator:\n    def __init__(self, val=0):\n        self.val = val\n    def __iadd__(self, other):\n        self.val += other\n        return self\n    def __repr__(self):\n        return f'Acc({self.val})'",
                "a = Accumulator()\na += 10\na += 20\nprint(f'a={a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=Acc(30)" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOperatorOverloadInteraction:
    """Test operator overloading patterns with cache invalidation."""

    def test_add_mul_overload_edit(self, nb_runner):
        """Editing vector class with __add__ and __mul__ should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Vec:\n"
                    "    def __init__(self, x, y):\n"
                    "        self.x = x\n"
                    "        self.y = y\n"
                    "    def __add__(self, other):\n"
                    "        return Vec(self.x + other.x, self.y + other.y)\n"
                    "    def __mul__(self, scalar):\n"
                    "        return Vec(self.x * scalar, self.y * scalar)\n"
                    "    def __repr__(self):\n"
                    "        return f'Vec({self.x},{self.y})'"
                ),
                "a = Vec(1, 2)\nb = Vec(3, 4)",
                "c = a + b\nd = c * 2",
                "print(f'c={c},d={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "c=Vec(4,6)" in out
        assert "d=Vec(8,12)" in out

        nb_runner.set_cell_source(2, "a = Vec(10, 20)\nb = Vec(30, 40)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "c=Vec(40,60)" in out
        assert "d=Vec(80,120)" in out

    def test_comparison_overload_edit(self, nb_runner):
        """Editing comparison overloads should propagate sorting."""
        nb_runner.create_notebook(
            [
                (
                    "class Score:\n"
                    "    def __init__(self, name, val):\n"
                    "        self.name = name\n"
                    "        self.val = val\n"
                    "    def __lt__(self, other):\n"
                    "        return self.val < other.val\n"
                    "    def __repr__(self):\n"
                    "        return f'{self.name}:{self.val}'"
                ),
                "scores = [Score('A', 30), Score('B', 10), Score('C', 20)]",
                "ranked = sorted(scores)",
                "result = ','.join(str(s) for s in ranked)",
                "print(f'ranked={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "ranked=B:10,C:20,A:30" in out

        nb_runner.set_cell_source(2, "scores = [Score('A', 5), Score('B', 50), Score('C', 25)]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "ranked=A:5,C:25,B:50" in out

    def test_contains_overload_edit(self, nb_runner):
        """Editing a class with __contains__ should propagate membership checks."""
        nb_runner.create_notebook(
            [
                (
                    "class WordSet:\n"
                    "    def __init__(self, words):\n"
                    "        self.words = set(w.lower() for w in words)\n"
                    "    def __contains__(self, item):\n"
                    "        return item.lower() in self.words"
                ),
                "ws = WordSet(['Hello', 'World'])",
                "checks = ['hello' in ws, 'python' in ws, 'WORLD' in ws]",
                "print(f'checks={checks}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "checks=[True, False, True]" in out

        nb_runner.set_cell_source(2, "ws = WordSet(['Python', 'Code'])")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "checks=[False, True, False]" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestChainedComparison:
    """chained comparison and identity operators."""

    def test_chained_compare(self, nb_runner):
        nb_runner.create_notebook(
            [
                "x = 5",
                "in_range = 1 < x < 10\nresult = 'yes' if in_range else 'no'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=yes" in nb_runner.get_output(2)

    def test_chained_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a, b, c = 1, 2, 3",
                "ascending = a < b < c\nresult = 'asc' if ascending else 'not'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=asc" in nb_runner.get_output(2)
        # Edit to break chain
        nb_runner.set_cell_source(1, "a, b, c = 1, 5, 3")
        nb_runner.run_all()
        assert "result=not" in nb_runner.get_output(2)

    def test_is_none_identity(self, nb_runner):
        nb_runner.create_notebook(
            [
                "val = None\nother = 0\nempty = ''",
                "r1 = val is None\nr2 = other is None\nr3 = empty is not None\nprint(f'r1={r1} r2={r2} r3={r3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=True r2=False r3=True" in nb_runner.get_output(2)
