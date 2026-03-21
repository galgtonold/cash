"""Batch 66: Abstract base classes & mixins — cash caching with ABC patterns."""
import textwrap
import pytest


@pytest.mark.stress
class TestABCPatterns:
    """Test abstract base class patterns across cells."""

    def test_abc_with_abstract_methods(self, nb_runner):
        """ABC with abstract methods implemented in subclasses."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from abc import ABC, abstractmethod

                class Shape(ABC):
                    @abstractmethod
                    def area(self) -> float: ...
                    @abstractmethod
                    def perimeter(self) -> float: ...
                    def describe(self) -> str:
                        return f"{self.__class__.__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}"

                class Circle(Shape):
                    def __init__(self, r):
                        self.r = r
                    def area(self):
                        import math
                        return math.pi * self.r ** 2
                    def perimeter(self):
                        import math
                        return 2 * math.pi * self.r

                class Rect(Shape):
                    def __init__(self, w, h):
                        self.w, self.h = w, h
                    def area(self):
                        return self.w * self.h
                    def perimeter(self):
                        return 2 * (self.w + self.h)

                shapes = [Circle(5), Rect(3, 4), Circle(10)]
            """),
            textwrap.dedent("""\
                descriptions = [s.describe() for s in shapes]
                for d in descriptions:
                    print(d)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Circle:" in out
        assert "Rect:" in out
        assert "area=12.00" in out  # 3*4

    def test_abc_register(self, nb_runner):
        """ABC virtual subclass registration."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from abc import ABC, abstractmethod

                class Serializable(ABC):
                    @abstractmethod
                    def to_dict(self): ...

                class MyData:
                    def __init__(self, val):
                        self.val = val
                    def to_dict(self):
                        return {'val': self.val}

                Serializable.register(MyData)
                d = MyData(42)
                print(f"is_serializable={isinstance(d, Serializable)}")
            """),
            textwrap.dedent("""\
                result = d.to_dict()
                print(f"dict={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "is_serializable=True" in nb_runner.get_output(1)
        assert "dict={'val': 42}" in nb_runner.get_output(2)


@pytest.mark.stress
class TestMixinPatterns:
    """Test mixin patterns across cells."""

    def test_mixin_composition(self, nb_runner):
        """Multiple mixins composed into a class."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class JsonMixin:
                    def to_json(self):
                        import json
                        data = {k: v for k, v in vars(self).items() if not k.startswith('_cash')}
                        return json.dumps(data)

                class ValidateMixin:
                    def validate(self):
                        for k, v in vars(self).items():
                            if k.startswith('_cash'):
                                continue
                            if v is None:
                                return False
                        return True

                class User(JsonMixin, ValidateMixin):
                    def __init__(self, name, email):
                        self.name = name
                        self.email = email

                u = User('Alice', 'alice@test.com')
                print(f"valid={u.validate()}")
                print(f"json={u.to_json()}")
            """),
            textwrap.dedent("""\
                u2 = User('Bob', None)
                print(f"u2_valid={u2.validate()}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid=True" in nb_runner.get_output(1)
        assert '"name": "Alice"' in nb_runner.get_output(1)
        assert "u2_valid=False" in nb_runner.get_output(2)

    def test_mixin_change_propagation(self, nb_runner):
        """Mixin behavior change propagates downstream."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class FormatMixin:
                    sep = ', '
                    def format_items(self, items):
                        return self.sep.join(str(i) for i in items)

                class Report(FormatMixin):
                    def __init__(self, data):
                        self.data = data
                    def summary(self):
                        return self.format_items(self.data)

                report = Report([1, 2, 3, 4, 5])
            """),
            textwrap.dedent("""\
                text = report.summary()
                print(f"summary={text}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary=1, 2, 3, 4, 5" in nb_runner.get_output(2)

        # Change separator
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class FormatMixin:
                sep = ' | '
                def format_items(self, items):
                    return self.sep.join(str(i) for i in items)

            class Report(FormatMixin):
                def __init__(self, data):
                    self.data = data
                def summary(self):
                    return self.format_items(self.data)

            report = Report([1, 2, 3, 4, 5])
        """))
        nb_runner.run_cells([1, 2])
        assert "summary=1 | 2 | 3 | 4 | 5" in nb_runner.get_output(2)
