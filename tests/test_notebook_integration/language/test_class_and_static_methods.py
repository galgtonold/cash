"""classmethod, staticmethod and class-level features edited across cells."""

import pytest


# class method / static method patterns with caching.
# Tests @classmethod, @staticmethod, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClassStaticMethod:
    """Test classmethod and staticmethod caching."""

    def test_classmethod_factory(self, nb_runner):
        """classmethod as factory with caching."""
        nb_runner.create_notebook(
            [
                "class Date:\n    def __init__(self, y, m, d):\n        self.y = y\n        self.m = m\n        self.d = d\n    @classmethod\n    def from_string(cls, s):\n        y, m, d = map(int, s.split('-'))\n        return cls(y, m, d)\n    def __str__(self):\n        return f'{self.y}/{self.m}/{self.d}'",
                "d = Date.from_string('2024-06-15')",
                "print(f'date={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "date=2024/6/15" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "date=2024/6/15" in out2

    def test_staticmethod_edit(self, nb_runner):
        """staticmethod with edit propagation."""
        nb_runner.create_notebook(
            [
                "class MathHelper:\n    @staticmethod\n    def clamp(val, lo, hi):\n        return max(lo, min(hi, val))",
                "val = 150",
                "result = MathHelper.clamp(val, 0, 100)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result=100" in out

        nb_runner.set_cell_source(2, "val = 50")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result=50" in out2

    def test_classmethod_counter(self, nb_runner):
        """classmethod tracking instance count."""
        nb_runner.create_notebook(
            [
                "class Widget:\n    _count = 0\n    def __init__(self, name):\n        self.name = name\n        Widget._count += 1\n    @classmethod\n    def get_count(cls):\n        return cls._count",
                "w1 = Widget('A')\nw2 = Widget('B')\nw3 = Widget('C')\ncount = Widget.get_count()",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=3" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "count=3" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClassMethodStaticMethod:
    """class method and static method patterns."""

    def test_classmethod_factory(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup cell",
                "class Date:\n    def __init__(self, y, m, d): self.y, self.m, self.d = y, m, d\n    @classmethod\n    def from_string(cls, s):\n        y, m, d = map(int, s.split('-'))\n        return cls(y, m, d)\n    def __str__(self): return f'{self.y}/{self.m}/{self.d}'\nd = Date.from_string('2024-01-15')\nprint(f'date={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "date=2024/1/15" in nb_runner.get_output(2)

    def test_staticmethod_util(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup cell",
                "class MathUtils:\n    @staticmethod\n    def is_prime(n):\n        if n < 2: return False\n        return all(n % i != 0 for i in range(2, int(n**0.5)+1))\nprimes = [n for n in range(20) if MathUtils.is_prime(n)]\nprint(f'primes={primes}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "primes=[2, 3, 5, 7, 11, 13, 17, 19]" in nb_runner.get_output(2)

    def test_classmethod_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup cell",
                "class Counter:\n    count = 0\n    @classmethod\n    def increment(cls): cls.count += 1\n    @classmethod\n    def get(cls): return cls.count\nCounter.increment()\nCounter.increment()\nprint(f'count={Counter.get()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "class Counter:\n    count = 0\n    @classmethod\n    def increment(cls): cls.count += 1\n    @classmethod\n    def get(cls): return cls.count\nfor _ in range(5): Counter.increment()\nprint(f'count={Counter.get()}')",
        )
        nb_runner.run_all()
        assert "count=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStaticClassMethods:
    """static methods and class methods."""

    def test_classmethod(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Date:\n    def __init__(self, y, m, d): self.y, self.m, self.d = y, m, d\n    @classmethod\n    def from_string(cls, s):\n        parts = s.split('-')\n        return cls(int(parts[0]), int(parts[1]), int(parts[2]))",
                "d = Date.from_string('2024-06-15')\nprint(f'y={d.y} m={d.m} d={d.d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y=2024" in nb_runner.get_output(2)
        assert "m=6" in nb_runner.get_output(2)
        assert "d=15" in nb_runner.get_output(2)

    def test_staticmethod(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Math:\n    @staticmethod\n    def clamp(val, lo, hi):\n        return max(lo, min(val, hi))",
                "r1 = Math.clamp(5, 0, 10)\nr2 = Math.clamp(-5, 0, 10)\nr3 = Math.clamp(15, 0, 10)\nprint(f'r1={r1} r2={r2} r3={r3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=5" in nb_runner.get_output(2)
        assert "r2=0" in nb_runner.get_output(2)
        assert "r3=10" in nb_runner.get_output(2)

    def test_classmethod_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Config:\n    def __init__(self, **kw): self.data = kw\n    @classmethod\n    def default(cls): return cls(mode='fast', debug=False)",
                'c = Config.default()\nprint(f\'mode={c.data["mode"]} debug={c.data["debug"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mode=fast" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            1,
            "class Config:\n    def __init__(self, **kw): self.data = kw\n    @classmethod\n    def default(cls): return cls(mode='slow', debug=True)",
        )
        nb_runner.run_all()
        assert "mode=slow" in nb_runner.get_output(2)
        assert "debug=True" in nb_runner.get_output(2)


# Property/classmethod/staticmethod interaction tests.
#
# Tests editing cells containing class features like properties,
# classmethods, and staticmethods to verify cache invalidation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestClassFeatureEdits:
    """Editing class properties, classmethods, and staticmethods."""

    def test_edit_staticmethod(self, nb_runner):
        """Edit a staticmethod and verify downstream uses new version."""
        nb_runner.create_notebook(
            [
                "class MathHelper:\n    @staticmethod\n    def double(x):\n        return x * 2",
                "result = MathHelper.double(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit to triple instead of double
        nb_runner.set_cell_source(1, "class MathHelper:\n    @staticmethod\n    def double(x):\n        return x * 3")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

    def test_edit_classmethod_factory(self, nb_runner):
        """Edit a classmethod factory and verify downstream."""
        nb_runner.create_notebook(
            [
                "class Config:\n    def __init__(self, val):\n        self.val = val\n    @classmethod\n    def default(cls):\n        return cls(42)",
                "cfg = Config.default()\nprint(f'val = {cfg.val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Change default value
        nb_runner.set_cell_source(
            1,
            "class Config:\n    def __init__(self, val):\n        self.val = val\n    @classmethod\n    def default(cls):\n        return cls(99)",
        )
        nb_runner.run_all()
        assert "val = 99" in nb_runner.get_output(2)

    def test_edit_property_getter(self, nb_runner):
        """Edit a property getter and verify downstream."""
        nb_runner.create_notebook(
            [
                "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
                "b = Box(3, 4)\nprint(f'area = {b.area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 12" in nb_runner.get_output(2)

        # Change to perimeter
        nb_runner.set_cell_source(
            1,
            "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return 2 * (self.w + self.h)",
        )
        nb_runner.run_all()
        assert "area = 14" in nb_runner.get_output(2)

    def test_edit_dunder_repr(self, nb_runner):
        """Edit __repr__ and verify output changes."""
        nb_runner.create_notebook(
            [
                "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'({self.x}, {self.y})'",
                "p = Point(1, 2)\nprint(f'p = {p}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p = (1, 2)" in nb_runner.get_output(2)

        # Edit __repr__
        nb_runner.set_cell_source(
            1,
            "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'",
        )
        nb_runner.run_all()
        assert "p = Point(1, 2)" in nb_runner.get_output(2)
