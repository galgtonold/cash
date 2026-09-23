"""Properties, cached_property and descriptors across cells."""

import textwrap

import pytest


# Cached property and memoization interaction tests.
#
# Tests editing cells with memoization patterns
# and verifying correct cache invalidation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMemoizationEdits:
    """Editing memoization patterns."""

    def test_edit_memoized_function(self, nb_runner):
        """Edit a function that uses manual memoization."""
        nb_runner.create_notebook(
            [
                "def fib(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]",
                "result = fib(10)\nprint(f'fib(10) = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib(10) = 55" in nb_runner.get_output(2)

        # Change to call with different arg
        nb_runner.set_cell_source(2, "result = fib(15)\nprint(f'fib(15) = {result}')")
        nb_runner.run_all()
        assert "fib(15) = 610" in nb_runner.get_output(2)

    def test_edit_lru_cache_function(self, nb_runner):
        """Edit function using lru_cache."""
        nb_runner.create_notebook(
            [
                "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
                "result = factorial(5)\nprint(f'5! = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "5! = 120" in nb_runner.get_output(2)

        # Change call
        nb_runner.set_cell_source(2, "result = factorial(7)\nprint(f'7! = {result}')")
        nb_runner.run_all()
        assert "7! = 5040" in nb_runner.get_output(2)

    def test_edit_lookup_table(self, nb_runner):
        """Edit a precomputed lookup table."""
        nb_runner.create_notebook(
            [
                "squares = {i: i**2 for i in range(10)}",
                "vals = [squares[x] for x in [1, 3, 5, 7]]\nprint(f'vals = {vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals = [1, 9, 25, 49]" in nb_runner.get_output(2)

        # Change to cubes
        nb_runner.set_cell_source(1, "squares = {i: i**3 for i in range(10)}")
        nb_runner.run_all()
        assert "vals = [1, 27, 125, 343]" in nb_runner.get_output(2)


# Descriptor protocol interaction tests.
# Tests that editing descriptor-based attribute access logic
# properly invalidates downstream cells.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDescriptorInteraction:
    """Test descriptor protocol patterns with cache invalidation."""

    def test_property_descriptor_edit(self, nb_runner):
        """Editing a class with property descriptors should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Circle:\n"
                    "    def __init__(self, radius):\n"
                    "        self._radius = radius\n"
                    "    @property\n"
                    "    def area(self):\n"
                    "        return 3.14159 * self._radius ** 2"
                ),
                "c = Circle(5)",
                "a = round(c.area, 2)",
                "print(f'area={a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "area=78.54" in out

        nb_runner.set_cell_source(2, "c = Circle(10)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "area=314.16" in out

    def test_custom_descriptor_edit(self, nb_runner):
        """Editing a custom descriptor class should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Validator:\n"
                    "    def __init__(self, min_val, max_val):\n"
                    "        self.min_val = min_val\n"
                    "        self.max_val = max_val\n"
                    "    def __set_name__(self, owner, name):\n"
                    "        self.name = '_' + name\n"
                    "    def __get__(self, obj, objtype=None):\n"
                    "        return getattr(obj, self.name, None)\n"
                    "    def __set__(self, obj, value):\n"
                    "        if not (self.min_val <= value <= self.max_val):\n"
                    "            raise ValueError(f'{value} not in [{self.min_val},{self.max_val}]')\n"
                    "        setattr(obj, self.name, value)"
                ),
                (
                    "class Sensor:\n"
                    "    temperature = Validator(-50, 150)\n"
                    "    def __init__(self, temp):\n"
                    "        self.temperature = temp"
                ),
                "s = Sensor(25)\nval = s.temperature",
                "print(f'temp={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "temp=25" in out

        nb_runner.set_cell_source(3, "s = Sensor(99)\nval = s.temperature")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "temp=99" in out

    def test_cached_property_edit(self, nb_runner):
        """Editing data used by a cached_property should propagate."""
        nb_runner.create_notebook(
            [
                "from functools import cached_property",
                (
                    "class Stats:\n"
                    "    def __init__(self, data):\n"
                    "        self.data = data\n"
                    "    @cached_property\n"
                    "    def mean(self):\n"
                    "        return sum(self.data) / len(self.data)"
                ),
                "st = Stats([10, 20, 30])\nm = st.mean",
                "print(f'mean={m}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mean=20.0" in out

        nb_runner.set_cell_source(3, "st = Stats([100, 200, 300])\nm = st.mean")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mean=200.0" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDescriptorProtocol:
    """descriptor protocol __get__ __set__."""

    def test_validated_descriptor(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Positive:\n    def __init__(self, name): self.name = name\n    def __set_name__(self, owner, name): self.name = name\n    def __get__(self, obj, objtype=None):\n        return getattr(obj, f'_{self.name}', 0)\n    def __set__(self, obj, value):\n        if value < 0: raise ValueError\n        setattr(obj, f'_{self.name}', value)\nclass Account:\n    balance = Positive('balance')\na = Account()\na.balance = 100\nprint(f'balance={a.balance}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "balance=100" in nb_runner.get_output(2)

    def test_cached_property_desc(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class CachedProp:\n    def __init__(self, fn): self.fn = fn; self.name = fn.__name__\n    def __get__(self, obj, objtype=None):\n        if obj is None: return self\n        val = self.fn(obj)\n        setattr(obj, self.name, val)\n        return val\nclass Data:\n    def __init__(self, n): self.n = n\n    @CachedProp\n    def expensive(self): return sum(range(self.n))\nd = Data(100)\nprint(f'result={d.expensive}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=4950" in nb_runner.get_output(2)

    def test_descriptor_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class TypedField:\n    def __init__(self, typ): self.typ = typ\n    def __set_name__(self, owner, name): self.name = f'_{name}'\n    def __get__(self, obj, t=None): return getattr(obj, self.name, None)\n    def __set__(self, obj, val):\n        if not isinstance(val, self.typ): raise TypeError\n        setattr(obj, self.name, val)\nclass Config:\n    port = TypedField(int)\nc = Config()\nc.port = 8080\nprint(f'port={c.port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "port=8080" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "class TypedField:\n    def __init__(self, typ): self.typ = typ\n    def __set_name__(self, owner, name): self.name = f'_{name}'\n    def __get__(self, obj, t=None): return getattr(obj, self.name, None)\n    def __set__(self, obj, val):\n        if not isinstance(val, self.typ): raise TypeError\n        setattr(obj, self.name, val)\nclass Config:\n    port = TypedField(int)\nc = Config()\nc.port = 3000\nprint(f'port={c.port}')",
        )
        nb_runner.run_all()
        assert "port=3000" in nb_runner.get_output(2)


# property decorator and computed attribute patterns with caching.
# Tests @property, computed fields, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPropertyComputed:
    """Test property decorator and computed attribute caching."""

    def test_property_basic(self, nb_runner):
        """Class with @property, verify caching."""
        nb_runner.create_notebook(
            [
                "class Rectangle:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
                "r = Rectangle(4, 5)",
                "result = r.area\nprint(f'area={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "area=20" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "area=20" in out2

    def test_property_edit_instance(self, nb_runner):
        """Edit instance creation, verify property updates."""
        nb_runner.create_notebook(
            [
                "class BMI:\n    def __init__(self, weight_kg, height_m):\n        self.weight = weight_kg\n        self.height = height_m\n    @property\n    def value(self):\n        return round(self.weight / self.height**2, 1)",
                "person = BMI(70, 1.75)",
                "bmi = person.value\nprint(f'bmi={bmi}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "bmi=22.9" in out

        nb_runner.set_cell_source(2, "person = BMI(90, 1.75)")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "bmi=29.4" in out2

    def test_property_with_setter(self, nb_runner):
        """Property with setter, verify caching."""
        nb_runner.create_notebook(
            [
                "class Temp:\n    def __init__(self, celsius):\n        self._c = celsius\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32",
                "t = Temp(100)\nf = t.fahrenheit",
                "print(f'f={f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "f=212" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "f=212" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPropertyComputedAttrs:
    """property decorators and computed attributes."""

    def test_property_getter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n    @property\n    def area(self):\n        import math\n        return round(math.pi * self._radius ** 2, 2)\n    @property\n    def circumference(self):\n        import math\n        return round(2 * math.pi * self._radius, 2)",
                "c = Circle(5)\nprint(f'area={c.area} circ={c.circumference}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=78.54" in nb_runner.get_output(2)
        assert "circ=31.42" in nb_runner.get_output(2)

    def test_property_setter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Temp:\n    def __init__(self, celsius):\n        self._c = celsius\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32\n    @fahrenheit.setter\n    def fahrenheit(self, f):\n        self._c = (f - 32) * 5/9",
                "t = Temp(100)\nf1 = t.fahrenheit\nt.fahrenheit = 32\nc1 = t._c\nprint(f'f1={f1} c1={c1}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f1=212.0" in nb_runner.get_output(2)
        assert "c1=0.0" in nb_runner.get_output(2)

    def test_property_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
                "b = Box(3, 4)\nprint(f'area={b.area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            1,
            "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
        )
        nb_runner.set_cell_source(2, "b = Box(10, 5)\nprint(f'area={b.area}')")
        nb_runner.run_all()
        assert "area=50" in nb_runner.get_output(2)


# Interaction test: property with computed cache and validation.
# Tests @property for computed values with internal caching,
# setter validation, and cross-cell attribute management.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPropertyComputedValidation:
    """Test property with computed values and validation across cells."""

    def test_property_computed(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: class with validated property
                "class BankAccount:\n    def __init__(self, owner, balance=0):\n        self.owner = owner\n        self._balance = balance\n    @property\n    def balance(self):\n        return self._balance\n    @balance.setter\n    def balance(self, value):\n        if value < 0:\n            raise ValueError('Negative balance')\n        self._balance = value\n    @property\n    def status(self):\n        if self._balance > 1000:\n            return 'premium'\n        elif self._balance > 0:\n            return 'active'\n        return 'empty'\n\nacc = BankAccount('Alice', 500)\nprint(f'balance={acc.balance}')\nprint(f'status={acc.status}')",
                # Cell 2: modify through property
                "acc.balance = 1500\nprint(f'new_balance={acc.balance}')\nprint(f'new_status={acc.status}')",
                # Cell 3: read state
                "info = f'{acc.owner}: ${acc.balance} ({acc.status})'\nprint(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "balance=500" in out1
        assert "status=active" in out1
        out2 = nb_runner.get_output(2)
        assert "new_balance=1500" in out2
        assert "new_status=premium" in out2
        out3 = nb_runner.get_output(3)
        assert "info=Alice: $1500 (premium)" in out3

    def test_property_computed_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Circle:\n    def __init__(self, radius):\n        self._r = radius\n    @property\n    def diameter(self):\n        return self._r * 2\n    @property\n    def circumference(self):\n        import math\n        return 2 * math.pi * self._r\n\nc = Circle(5)\nprint(f'diam={c.diameter}')\nprint(f'circ={c.circumference:.2f}')",
                "ratio = c.circumference / c.diameter\nprint(f'ratio={ratio:.4f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "diam=10" in nb_runner.get_output(1)
        assert "ratio=3.1416" in nb_runner.get_output(2)

        # Edit radius
        nb_runner.set_cell_source(
            1,
            "class Circle:\n    def __init__(self, radius):\n        self._r = radius\n    @property\n    def diameter(self):\n        return self._r * 2\n    @property\n    def circumference(self):\n        import math\n        return 2 * math.pi * self._r\n\nc = Circle(10)\nprint(f'diam={c.diameter}')\nprint(f'circ={c.circumference:.2f}')",
        )
        nb_runner.run_cells([1, 2])
        assert "diam=20" in nb_runner.get_output(1)
        assert "ratio=3.1416" in nb_runner.get_output(2)

    def test_property_computed_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Config:\n    def __init__(self, values):\n        self._values = values\n    @property\n    def count(self):\n        return len(self._values)\n    @property\n    def summary(self):\n        return f'{self.count} items'\n\ncfg = Config([1, 2, 3])\nprint(f'summary={cfg.summary}')",
                "has_many = cfg.count > 2\nprint(f'has_many={has_many}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary=3 items" in nb_runner.get_output(1)
        assert "has_many=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_many=True" in nb_runner.get_output(2)


# Interaction test: property with deleter and validation.
# Tests property getter/setter/deleter with validation logic,
# AttributeError handling, and cross-cell state management.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPropertyDeleterValidation:
    """Test property with deleter and validation across cells."""

    def test_property_validation(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define class with validated property
                "class Temperature:\n    def __init__(self, celsius):\n        self.celsius = celsius\n    @property\n    def celsius(self):\n        return self._celsius\n    @celsius.setter\n    def celsius(self, value):\n        if value < -273.15:\n            raise ValueError('Below absolute zero')\n        self._celsius = value\n    @celsius.deleter\n    def celsius(self):\n        self._celsius = 0.0\n    @property\n    def fahrenheit(self):\n        return self._celsius * 9/5 + 32\nprint('Temperature defined')",
                # Cell 2: use property
                "t = Temperature(100)\nprint(f'c={t.celsius}')\nprint(f'f={t.fahrenheit}')\nt.celsius = 0\nprint(f'freezing_f={t.fahrenheit}')",
                # Cell 3: deleter and validation
                "del t.celsius\nprint(f'after_del={t.celsius}')\ntry:\n    t.celsius = -300\n    print('no_error')\nexcept ValueError as e:\n    print(f'error={e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "c=100" in out2
        assert "f=212.0" in out2
        assert "freezing_f=32.0" in out2
        out3 = nb_runner.get_output(3)
        assert "after_del=0.0" in out3
        assert "error=Below absolute zero" in out3

    def test_property_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h\nprint('Box defined')",
                "b = Box(5, 3)\nprint(f'area={b.area}')",
                "double_area = b.area * 2\nprint(f'double={double_area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=15" in nb_runner.get_output(2)
        assert "double=30" in nb_runner.get_output(3)

        # Edit box dimensions
        nb_runner.set_cell_source(2, "b = Box(10, 7)\nprint(f'area={b.area}')")
        nb_runner.run_cells([2, 3])
        assert "area=70" in nb_runner.get_output(2)
        assert "double=140" in nb_runner.get_output(3)

    def test_property_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def radius(self):\n        return self._r\n    @radius.setter\n    def radius(self, val):\n        if val < 0:\n            raise ValueError('Negative')\n        self._r = val\nprint('Circle defined')",
                "c = Circle(5)\nprint(f'r={c.radius}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "r=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPropertyGetterSetter:
    """property decorators getters and setters."""

    def test_property_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Circle:\n    def __init__(self, r): self._r = r\n    @property\n    def radius(self): return self._r\n    @radius.setter\n    def radius(self, val):\n        if val < 0: raise ValueError\n        self._r = val\n    @property\n    def area(self): return 3.14159 * self._r ** 2",
                "c = Circle(5)\na1 = round(c.area, 2)\nc.radius = 10\na2 = round(c.area, 2)\nprint(f'a1={a1} a2={a2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a1=78.54" in nb_runner.get_output(2)
        assert "a2=314.16" in nb_runner.get_output(2)

    def test_property_validation(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Temp:\n    def __init__(self, c): self._c = c\n    @property\n    def fahrenheit(self): return self._c * 9/5 + 32\n    @property\n    def celsius(self): return self._c",
                "t = Temp(100)\nprint(f'c={t.celsius} f={t.fahrenheit}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c=100" in nb_runner.get_output(2)
        assert "f=212.0" in nb_runner.get_output(2)

    def test_property_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Box:\n    def __init__(self, w, h): self.w, self.h = w, h\n    @property\n    def area(self): return self.w * self.h",
                "b = Box(3, 4)\nprint(f'area={b.area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "b = Box(10, 20)\nprint(f'area={b.area}')")
        nb_runner.run_all()
        assert "area=200" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPropertySetterDeleter:
    """class property setter/deleter and computed attrs."""

    def test_property_getter_setter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Temperature:\n    def __init__(self, celsius):\n        self._celsius = celsius\n    @property\n    def fahrenheit(self):\n        return self._celsius * 9/5 + 32\n    @fahrenheit.setter\n    def fahrenheit(self, value):\n        self._celsius = (value - 32) * 5/9",
                "t = Temperature(100)\nf = t.fahrenheit\nt.fahrenheit = 32\nc = t._celsius\nprint(f'f={f} c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f=212.0" in nb_runner.get_output(2)
        assert "c=0.0" in nb_runner.get_output(2)

    def test_property_edit_class(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Circle:\n    def __init__(self, radius):\n        self.radius = radius\n    @property\n    def area(self):\n        import math\n        return round(math.pi * self.radius ** 2, 2)",
                "c = Circle(5)\nresult = c.area\nprint(f'area={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=78.54" in nb_runner.get_output(2)
        # Edit to add circumference
        nb_runner.set_cell_source(
            1,
            "class Circle:\n    def __init__(self, radius):\n        self.radius = radius\n    @property\n    def area(self):\n        import math\n        return round(math.pi * self.radius ** 2, 2)\n    @property\n    def circumference(self):\n        import math\n        return round(2 * math.pi * self.radius, 2)",
        )
        nb_runner.set_cell_source(2, "c = Circle(5)\nresult = c.circumference\nprint(f'circ={result}')")
        nb_runner.run_all()
        assert "circ=31.42" in nb_runner.get_output(2)

    def test_property_validation(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Age:\n    def __init__(self, value):\n        self.value = value\n    @property\n    def value(self):\n        return self._value\n    @value.setter\n    def value(self, v):\n        self._value = max(0, min(150, v))",
                "a = Age(200)\nresult = a.value\nprint(f'clamped={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "clamped=150" in nb_runner.get_output(2)


# Descriptor, property, slots, dataclass, and protocol patterns.
class TestDescriptorPatterns:
    """Test caching with Python descriptors."""

    @pytest.mark.integration
    @pytest.mark.stress
    def test_property_change_class(self, nb_runner):
        """Change property logic → downstream updates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Rect:
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                    
                    @property
                    def area(self):
                        return self.w * self.h
            """),
                "r = Rect(4, 5)",
                "print(r.area)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        # Change property to include perimeter
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Rect:
                def __init__(self, w, h):
                    self.w = w
                    self.h = h
                
                @property
                def area(self):
                    return self.w * self.h
                
                @property
                def perimeter(self):
                    return 2 * (self.w + self.h)
        """),
        )
        nb_runner.set_cell_source(3, "print(f'{r.area} {r.perimeter}')")
        nb_runner.run_all()
        assert "20 18" in nb_runner.get_output(3)

    # Property & descriptor patterns — cash caching with properties and descriptors.
    @pytest.mark.stress
    def test_type_checked_descriptor(self, nb_runner):
        """Type-checking descriptor across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class TypeChecked:
                    def __init__(self, expected_type):
                        self.expected_type = expected_type
                    def __set_name__(self, owner, name):
                        self.name = name
                        self.attr = f'_tc_{name}'
                    def __get__(self, obj, objtype=None):
                        if obj is None:
                            return self
                        return getattr(obj, self.attr, None)
                    def __set__(self, obj, value):
                        if not isinstance(value, self.expected_type):
                            raise TypeError(f"{self.name} must be {self.expected_type.__name__}")
                        setattr(obj, self.attr, value)

                class Config:
                    host = TypeChecked(str)
                    port = TypeChecked(int)
                    debug = TypeChecked(bool)
                    def __init__(self, host, port, debug=False):
                        self.host = host
                        self.port = port
                        self.debug = debug

                cfg = Config('localhost', 8080, True)
                print(f"host={cfg.host} port={cfg.port} debug={cfg.debug}")
            """),
                textwrap.dedent("""\
                url = f"http://{cfg.host}:{cfg.port}"
                print(f"url={url}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost port=8080 debug=True" in nb_runner.get_output(1)
        assert "url=http://localhost:8080" in nb_runner.get_output(2)

    @pytest.mark.stress
    def test_property_propagation(self, nb_runner):
        """Property class propagation on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Box:
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                    @property
                    def area(self):
                        return self.w * self.h

                box = Box(5, 10)
            """),
                textwrap.dedent("""\
                print(f"area={box.area}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=50" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Box:
                def __init__(self, w, h):
                    self.w = w
                    self.h = h
                @property
                def area(self):
                    return self.w * self.h

            box = Box(8, 12)
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "area=96" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestContextManagerPatterns:
    """Test caching with context managers."""

    def test_custom_context_manager_class(self, nb_runner):
        """Custom __enter__/__exit__ context manager."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Timer:
                    def __enter__(self):
                        import time
                        self.start = time.time()
                        return self
                    def __exit__(self, *args):
                        import time
                        self.elapsed = time.time() - self.start
            """),
                textwrap.dedent("""\
                import time
                with Timer() as t:
                    time.sleep(0.01)
                print(f"elapsed={t.elapsed > 0}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "elapsed=True" in nb_runner.get_output(2)
