"""Batch 81: Property & descriptor patterns — cash caching with properties and descriptors."""
import textwrap
import pytest


@pytest.mark.stress
class TestPropertyPatterns:
    """Test property patterns across cells."""

    def test_computed_property(self, nb_runner):
        """Computed property across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Circle:
                    def __init__(self, radius):
                        self._radius = radius

                    @property
                    def radius(self):
                        return self._radius

                    @radius.setter
                    def radius(self, value):
                        if value < 0:
                            raise ValueError("Radius must be non-negative")
                        self._radius = value

                    @property
                    def area(self):
                        import math
                        return math.pi * self._radius ** 2

                    @property
                    def circumference(self):
                        import math
                        return 2 * math.pi * self._radius

                c = Circle(5)
                print(f"r={c.radius} area={c.area:.2f} circ={c.circumference:.2f}")
            """),
            textwrap.dedent("""\
                c.radius = 10
                print(f"new_area={c.area:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=5 area=78.54" in nb_runner.get_output(1)
        assert "new_area=314.16" in nb_runner.get_output(2)

    def test_cached_property(self, nb_runner):
        """functools.cached_property pattern across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class ExpensiveCompute:
                    def __init__(self, data):
                        self.data = data
                        self._compute_count = 0

                    @property
                    def result(self):
                        self._compute_count += 1
                        return sum(x**2 for x in self.data)

                obj = ExpensiveCompute([1, 2, 3, 4, 5])
                r1 = obj.result
                r2 = obj.result
                print(f"r1={r1} r2={r2} computes={obj._compute_count}")
            """),
            textwrap.dedent("""\
                print(f"result={obj.result} total_computes={obj._compute_count}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=55 r2=55" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "result=55" in out2


@pytest.mark.stress
class TestDescriptorPatterns:
    """Test descriptor protocol patterns across cells."""

    def test_validated_descriptor(self, nb_runner):
        """Descriptor with validation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Positive:
                    def __init__(self, name):
                        self.name = name
                    def __set_name__(self, owner, name):
                        self.attr = f'_desc_{name}'
                    def __get__(self, obj, objtype=None):
                        return getattr(obj, self.attr, None)
                    def __set__(self, obj, value):
                        if value < 0:
                            raise ValueError(f"{self.name} must be positive")
                        setattr(obj, self.attr, value)

                class Product:
                    price = Positive('price')
                    quantity = Positive('quantity')
                    def __init__(self, name, price, qty):
                        self.name = name
                        self.price = price
                        self.quantity = qty

                p = Product('Widget', 9.99, 100)
                print(f"product={p.name} price={p.price} qty={p.quantity}")
            """),
            textwrap.dedent("""\
                total = p.price * p.quantity
                print(f"total=${total:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "product=Widget price=9.99 qty=100" in nb_runner.get_output(1)
        assert "total=$999.00" in nb_runner.get_output(2)

    def test_type_checked_descriptor(self, nb_runner):
        """Type-checking descriptor across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost port=8080 debug=True" in nb_runner.get_output(1)
        assert "url=http://localhost:8080" in nb_runner.get_output(2)

    def test_property_propagation(self, nb_runner):
        """Property class propagation on change."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=50" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class Box:
                def __init__(self, w, h):
                    self.w = w
                    self.h = h
                @property
                def area(self):
                    return self.w * self.h

            box = Box(8, 12)
        """))
        nb_runner.run_cells([1, 2])
        assert "area=96" in nb_runner.get_output(2)
