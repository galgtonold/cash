"""Batch 47: Complex inheritance & MRO patterns — diamond, mixin, super() chains."""
import textwrap
import pytest


@pytest.mark.stress
class TestDiamondInheritance:
    """Test diamond inheritance and MRO."""

    def test_diamond_mro(self, nb_runner):
        """Classic diamond inheritance with super() chain."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    def who(self):
                        return ['Base']

                class Left(Base):
                    def who(self):
                        return ['Left'] + super().who()

                class Right(Base):
                    def who(self):
                        return ['Right'] + super().who()

                class Diamond(Left, Right):
                    def who(self):
                        return ['Diamond'] + super().who()
            """),
            textwrap.dedent("""\
                d = Diamond()
                chain = d.who()
                print(f"chain={chain}")
                mro = [c.__name__ for c in Diamond.__mro__]
                print(f"mro={mro}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "chain=['Diamond', 'Left', 'Right', 'Base']" in nb_runner.get_output(2)
        assert "'Diamond', 'Left', 'Right', 'Base', 'object'" in nb_runner.get_output(2)

    def test_diamond_change_base(self, nb_runner):
        """Changing base class propagates through diamond."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    value = 10
                class Left(Base): pass
                class Right(Base): pass
                class Diamond(Left, Right): pass
            """),
            textwrap.dedent("""\
                result = Diamond.value
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)

        # Change base
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class Base:
                value = 99
            class Left(Base): pass
            class Right(Base): pass
            class Diamond(Left, Right): pass
        """))
        nb_runner.run_all()
        assert "result=99" in nb_runner.get_output(2)


@pytest.mark.stress
class TestMixinPatterns:
    """Test mixin class patterns."""

    def test_multiple_mixins(self, nb_runner):
        """Multiple mixins providing different features."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class JsonMixin:
                    def to_json(self):
                        import json
                        d = {k: v for k, v in self.__dict__.items() if not k.startswith('_cash')}
                        return json.dumps(d, sort_keys=True)

                class LogMixin:
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._log = []
                    def log(self, msg):
                        self._log.append(msg)

                class ValidateMixin:
                    def validate(self):
                        for k, v in self.__dict__.items():
                            if k.startswith('_'):
                                continue
                            if v is None:
                                return False
                        return True
            """),
            textwrap.dedent("""\
                class User(JsonMixin, LogMixin, ValidateMixin):
                    def __init__(self, name, email):
                        super().__init__()
                        self.name = name
                        self.email = email

                u = User("Alice", "alice@example.com")
                u.log("created")
                json_out = u.to_json()
                valid = u.validate()
                print(f"json={json_out}")
                print(f"valid={valid} log_count={len(u._log)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert '"name": "Alice"' in out
        assert "valid=True" in out

    def test_mixin_evolution(self, nb_runner):
        """Evolving mixin adds new method."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class PrintMixin:
                    def describe(self):
                        return f"Object with {len(self.__dict__)} attrs"
            """),
            textwrap.dedent("""\
                class Item(PrintMixin):
                    def __init__(self, name, price):
                        self.name = name
                        self.price = price

                item = Item("Widget", 9.99)
                desc = item.describe()
                print(f"desc={desc}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Object with" in out

        # Evolve mixin
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class PrintMixin:
                def describe(self):
                    attrs = ', '.join(f'{k}={v}' for k, v in sorted(self.__dict__.items()) if not k.startswith('_cash'))
                    return f"Object({attrs})"
        """))
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "name=Widget" in out2
        assert "price=9.99" in out2


@pytest.mark.stress
class TestAbstractPatterns:
    """Test abstract base class patterns."""

    def test_abc_enforcement(self, nb_runner):
        """ABC with abstract methods across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from abc import ABC, abstractmethod

                class Shape(ABC):
                    @abstractmethod
                    def area(self):
                        pass
                    @abstractmethod
                    def perimeter(self):
                        pass
            """),
            textwrap.dedent("""\
                class Circle(Shape):
                    def __init__(self, radius):
                        self.radius = radius
                    def area(self):
                        import math
                        return math.pi * self.radius ** 2
                    def perimeter(self):
                        import math
                        return 2 * math.pi * self.radius

                c = Circle(5)
                print(f"area={c.area():.2f} perim={c.perimeter():.2f}")
            """),
            textwrap.dedent("""\
                class Rectangle(Shape):
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                    def area(self):
                        return self.w * self.h
                    def perimeter(self):
                        return 2 * (self.w + self.h)

                shapes = [Circle(3), Rectangle(4, 5)]
                areas = [f"{s.area():.1f}" for s in shapes]
                print(f"areas={areas}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=78.54" in nb_runner.get_output(2)
        assert "28.3" in nb_runner.get_output(3)  # pi*9
        assert "20.0" in nb_runner.get_output(3)  # 4*5

    def test_super_init_chain(self, nb_runner):
        """Complex __init__ chain with super()."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class A:
                    def __init__(self, **kwargs):
                        self.a_val = kwargs.pop('a', 0)
                        super().__init__(**kwargs)

                class B(A):
                    def __init__(self, **kwargs):
                        self.b_val = kwargs.pop('b', 0)
                        super().__init__(**kwargs)

                class C(B):
                    def __init__(self, **kwargs):
                        self.c_val = kwargs.pop('c', 0)
                        super().__init__(**kwargs)
            """),
            textwrap.dedent("""\
                obj = C(a=1, b=2, c=3)
                print(f"a={obj.a_val} b={obj.b_val} c={obj.c_val}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=2 c=3" in nb_runner.get_output(2)
