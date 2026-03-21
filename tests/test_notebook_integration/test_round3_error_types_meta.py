"""
Batch 14: Error handling patterns, type annotations, abstract classes,
metaclass interactions, and exception flow caching.

Tests how cash handles try/except, custom exceptions, type-annotated code,
abstract base classes, metaclass-driven class creation, and exception
propagation across cells.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress]


# ============================================================
# Test Group 1: Error Handling Patterns
# ============================================================

class TestErrorHandlingPatterns:
    """Test try/except/finally patterns and their caching behavior."""

    def test_try_except_cached(self, nb_runner):
        """Try/except block with successful path should be cacheable."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                try:
                    result = int("42")
                except ValueError:
                    result = -1
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(1)

    def test_try_except_error_path(self, nb_runner):
        """Try/except catching an error — result from except branch."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                try:
                    result = int("not_a_number")
                except ValueError:
                    result = -999
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "-999" in nb_runner.get_output(1)

    def test_try_except_finally(self, nb_runner):
        """Try/except/finally — finally always runs."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                cleanup_ran = False
                try:
                    value = 100
                except Exception:
                    value = -1
                finally:
                    cleanup_ran = True
                print(value, cleanup_ran)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100 True" in nb_runner.get_output(1)

    def test_custom_exception_class(self, nb_runner):
        """Custom exception defined in one cell, raised/caught in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class AppError(Exception):
                    def __init__(self, code, msg):
                        self.code = code
                        self.msg = msg
                        super().__init__(msg)
            """),
            textwrap.dedent("""\
                try:
                    raise AppError(404, "Not Found")
                except AppError as e:
                    error_info = f"{e.code}: {e.msg}"
                print(error_info)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "404: Not Found" in nb_runner.get_output(2)

    def test_exception_chain(self, nb_runner):
        """Exception chaining with 'from'."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                try:
                    try:
                        x = 1 / 0
                    except ZeroDivisionError as e:
                        raise ValueError("bad input") from e
                except ValueError as e:
                    result = str(e)
                    cause = str(e.__cause__)
                print(result, "|", cause)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "bad input" in output
        assert "division by zero" in output

    def test_nested_try_except(self, nb_runner):
        """Nested try/except blocks."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                results = []
                for val in ["10", "abc", "20", "xyz"]:
                    try:
                        try:
                            results.append(int(val))
                        except ValueError:
                            results.append(float('nan'))
                    except Exception:
                        results.append(None)
                print(len(results), results[0], results[2])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "4" in output
        assert "10" in output
        assert "20" in output


# ============================================================
# Test Group 2: Type Annotations
# ============================================================

class TestTypeAnnotations:
    """Test that type-annotated code caches correctly."""

    def test_typed_function(self, nb_runner):
        """Function with type annotations."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def add(x: int, y: int) -> int:
                    return x + y

                result: int = add(3, 4)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(1)

    def test_typed_variable_annotations(self, nb_runner):
        """Variable annotations without assignment."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from typing import List, Dict, Optional

                scores: List[int] = [90, 85, 92]
                lookup: Dict[str, int] = {"a": 1, "b": 2}
                maybe: Optional[str] = None
                print(len(scores), len(lookup), maybe)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3 2 None" in nb_runner.get_output(1)

    def test_typed_class(self, nb_runner):
        """Class with typed attributes."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from dataclasses import dataclass

                @dataclass
                class Point:
                    x: float
                    y: float

                    def distance(self) -> float:
                        return (self.x**2 + self.y**2) ** 0.5

                p = Point(3.0, 4.0)
                print(f"{p.distance():.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "5.0" in nb_runner.get_output(1)

    def test_generic_type_alias(self, nb_runner):
        """Generic type aliases (Python 3.12+ style and older)."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from typing import TypeVar, Generic

                T = TypeVar('T')

                class Box(Generic[T]):
                    def __init__(self, content: T):
                        self.content = content
                    def get(self) -> T:
                        return self.content

                b = Box(42)
                print(b.get())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(1)


# ============================================================
# Test Group 3: Abstract Base Classes
# ============================================================

class TestAbstractBaseClasses:
    """Test ABC patterns and their caching behavior."""

    def test_abc_basic(self, nb_runner):
        """Basic ABC with abstract method."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from abc import ABC, abstractmethod

                class Shape(ABC):
                    @abstractmethod
                    def area(self) -> float:
                        pass
            """),
            textwrap.dedent("""\
                class Rectangle(Shape):
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                    def area(self) -> float:
                        return self.w * self.h

                r = Rectangle(3, 4)
                print(r.area())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "12" in nb_runner.get_output(2)

    def test_abc_cannot_instantiate(self, nb_runner):
        """Attempting to instantiate an ABC raises TypeError."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from abc import ABC, abstractmethod

                class Base(ABC):
                    @abstractmethod
                    def do(self):
                        pass
            """),
            textwrap.dedent("""\
                try:
                    b = Base()
                    msg = "no error"
                except TypeError:
                    msg = "TypeError caught"
                print(msg)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "TypeError caught" in nb_runner.get_output(2)

    def test_abc_multiple_abstract_methods(self, nb_runner):
        """ABC with multiple abstract methods implemented in stages."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from abc import ABC, abstractmethod

                class Serializable(ABC):
                    @abstractmethod
                    def serialize(self) -> str:
                        pass

                    @abstractmethod
                    def deserialize(self, data: str):
                        pass
            """),
            textwrap.dedent("""\
                class SimpleData(Serializable):
                    def __init__(self, value=0):
                        self.value = value
                    def serialize(self) -> str:
                        return str(self.value)
                    def deserialize(self, data: str):
                        self.value = int(data)
                        return self
            """),
            textwrap.dedent("""\
                sd = SimpleData(42)
                serialized = sd.serialize()
                sd2 = SimpleData().deserialize(serialized)
                print(sd2.value)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(3)


# ============================================================
# Test Group 4: Metaclass Interactions
# ============================================================

class TestMetaclassInteractions:
    """Test metaclass-driven class creation and caching."""

    def test_simple_metaclass(self, nb_runner):
        """Metaclass that modifies class creation."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class UpperAttrMeta(type):
                    def __new__(mcs, name, bases, namespace):
                        uppercase_attrs = {}
                        for key, val in namespace.items():
                            if not key.startswith('_'):
                                uppercase_attrs[key.upper()] = val
                            else:
                                uppercase_attrs[key] = val
                        return super().__new__(mcs, name, bases, uppercase_attrs)
            """),
            textwrap.dedent("""\
                class MyClass(metaclass=UpperAttrMeta):
                    greeting = "hello"
                    farewell = "goodbye"

                obj = MyClass()
                print(hasattr(obj, 'GREETING'), obj.GREETING)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "True" in output
        assert "hello" in output

    def test_singleton_metaclass(self, nb_runner):
        """Singleton pattern via metaclass."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class SingletonMeta(type):
                    _instances = {}
                    def __call__(cls, *args, **kwargs):
                        if cls not in cls._instances:
                            cls._instances[cls] = super().__call__(*args, **kwargs)
                        return cls._instances[cls]
            """),
            textwrap.dedent("""\
                class Database(metaclass=SingletonMeta):
                    def __init__(self):
                        self.connection = "active"

                db1 = Database()
                db2 = Database()
                print(db1 is db2, db1.connection)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "True active" in nb_runner.get_output(2)

    def test_registry_metaclass(self, nb_runner):
        """Metaclass that auto-registers subclasses."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class PluginMeta(type):
                    registry = {}
                    def __new__(mcs, name, bases, namespace):
                        cls = super().__new__(mcs, name, bases, namespace)
                        if bases:  # Don't register the base class
                            mcs.registry[name] = cls
                        return cls

                class Plugin(metaclass=PluginMeta):
                    pass
            """),
            textwrap.dedent("""\
                class AudioPlugin(Plugin):
                    kind = "audio"

                class VideoPlugin(Plugin):
                    kind = "video"

                print(sorted(PluginMeta.registry.keys()))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "AudioPlugin" in output
        assert "VideoPlugin" in output


# ============================================================
# Test Group 5: String Processing Patterns
# ============================================================

class TestStringProcessingPatterns:
    """Test complex string processing and regex patterns."""

    def test_regex_across_cells(self, nb_runner):
        """Regex pattern compiled in one cell, used in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import re
                email_pattern = re.compile(r'[\\w.]+@[\\w]+\\.[\\w]+')
            """),
            textwrap.dedent("""\
                text = "Contact alice@example.com or bob@test.org"
                emails = email_pattern.findall(text)
                print(emails)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "alice@example.com" in output
        assert "bob@test.org" in output

    def test_string_template(self, nb_runner):
        """String template pattern across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from string import Template
                tmpl = Template("Hello $name, you have $count items")
            """),
            textwrap.dedent("""\
                msg = tmpl.substitute(name="Alice", count=5)
                print(msg)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello Alice, you have 5 items" in nb_runner.get_output(2)

    def test_multiline_string_processing(self, nb_runner):
        """Processing multiline strings with splitlines."""
        nb_runner.create_notebook([
            textwrap.dedent('''\
                text = """line1
                line2
                line3
                line4"""
            '''),
            textwrap.dedent("""\
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                print(len(lines), lines[0], lines[-1])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "4" in output
        assert "line1" in output
        assert "line4" in output

    def test_json_processing_across_cells(self, nb_runner):
        """JSON encode in one cell, decode in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import json
                data = {"users": [{"name": "Alice", "age": 30}]}
                encoded = json.dumps(data)
            """),
            textwrap.dedent("""\
                decoded = json.loads(encoded)
                user_name = decoded['users'][0]['name']
                print(user_name)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice" in nb_runner.get_output(2)

    def test_format_spec_patterns(self, nb_runner):
        """Various format spec patterns."""
        nb_runner.create_notebook([
            "value = 3.14159265",
            textwrap.dedent("""\
                results = [
                    f"{value:.2f}",
                    f"{value:.4e}",
                    f"{1000000:,}",
                    f"{0.75:.1%}",
                ]
                print(results)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "3.14" in output
        assert "1,000,000" in output
        assert "75.0%" in output
