"""Batch 48: Metaclass patterns — cash caching with metaclasses and class hooks."""
import textwrap
import pytest


@pytest.mark.stress
class TestMetaclassBasics:
    """Test basic metaclass usage."""

    def test_simple_metaclass(self, nb_runner):
        """Metaclass that adds a class attribute."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class AutoRepr(type):
                    def __new__(mcs, name, bases, namespace):
                        cls = super().__new__(mcs, name, bases, namespace)
                        cls._class_name = name
                        return cls
            """),
            textwrap.dedent("""\
                class MyClass(metaclass=AutoRepr):
                    pass

                class SubClass(MyClass):
                    pass

                print(f"name1={MyClass._class_name} name2={SubClass._class_name}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name1=MyClass name2=SubClass" in nb_runner.get_output(2)

    def test_metaclass_registry(self, nb_runner):
        """Metaclass that maintains a registry."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Registry(type):
                    _registry = {}
                    def __new__(mcs, name, bases, namespace):
                        cls = super().__new__(mcs, name, bases, namespace)
                        if name != 'Base':
                            Registry._registry[name] = cls
                        return cls
            """),
            textwrap.dedent("""\
                class Base(metaclass=Registry):
                    pass

                class PluginA(Base):
                    value = 1

                class PluginB(Base):
                    value = 2
            """),
            textwrap.dedent("""\
                names = sorted(Registry._registry.keys())
                values = [Registry._registry[n].value for n in names]
                print(f"plugins={names} values={values}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "plugins=['PluginA', 'PluginB']" in nb_runner.get_output(3)
        assert "values=[1, 2]" in nb_runner.get_output(3)

    def test_metaclass_change_propagates(self, nb_runner):
        """Changing metaclass-created class propagates."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Versioned(type):
                    def __new__(mcs, name, bases, namespace):
                        cls = super().__new__(mcs, name, bases, namespace)
                        cls.version = namespace.get('VERSION', '0.0')
                        return cls
            """),
            textwrap.dedent("""\
                class API(metaclass=Versioned):
                    VERSION = '1.0'
                print(f"version={API.version}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "version=1.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, textwrap.dedent("""\
            class API(metaclass=Versioned):
                VERSION = '2.0'
            print(f"version={API.version}")
        """))
        nb_runner.run_all()
        assert "version=2.0" in nb_runner.get_output(2)


@pytest.mark.stress
class TestInitSubclass:
    """Test __init_subclass__ patterns."""

    def test_init_subclass_validation(self, nb_runner):
        """__init_subclass__ for validating subclasses."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Validated:
                    def __init_subclass__(cls, required_attr=None, **kwargs):
                        super().__init_subclass__(**kwargs)
                        if required_attr and not hasattr(cls, required_attr):
                            raise TypeError(f"{cls.__name__} must define {required_attr}")
                        cls._validated = True
            """),
            textwrap.dedent("""\
                class Good(Validated, required_attr='name'):
                    name = "good_class"

                print(f"validated={Good._validated} name={Good.name}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "validated=True name=good_class" in nb_runner.get_output(2)

    def test_init_subclass_auto_register(self, nb_runner):
        """__init_subclass__ for auto-registration."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Command:
                    _commands = {}
                    def __init_subclass__(cls, command_name=None, **kwargs):
                        super().__init_subclass__(**kwargs)
                        if command_name:
                            Command._commands[command_name] = cls
            """),
            textwrap.dedent("""\
                class ListCmd(Command, command_name='list'):
                    def run(self): return 'listing'

                class CreateCmd(Command, command_name='create'):
                    def run(self): return 'creating'
            """),
            textwrap.dedent("""\
                cmds = sorted(Command._commands.keys())
                results = [Command._commands[c]().run() for c in cmds]
                print(f"commands={cmds} results={results}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "commands=['create', 'list']" in nb_runner.get_output(3)
        assert "results=['creating', 'listing']" in nb_runner.get_output(3)


@pytest.mark.stress
class TestClassDecorators:
    """Test class-level decorators."""

    def test_class_decorator_singleton(self, nb_runner):
        """Class decorator implementing singleton pattern."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def singleton(cls):
                    instances = {}
                    def get_instance(*args, **kwargs):
                        if cls not in instances:
                            instances[cls] = cls(*args, **kwargs)
                        return instances[cls]
                    return get_instance
            """),
            textwrap.dedent("""\
                @singleton
                class Database:
                    def __init__(self, url='default'):
                        self.url = url

                db1 = Database('postgres://localhost')
                db2 = Database('other')  # ignored, already created
                print(f"same={db1 is db2} url={db1.url}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "same=True url=postgres://localhost" in nb_runner.get_output(2)

    def test_dataclass_like_decorator(self, nb_runner):
        """Decorator that adds methods like dataclass."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def auto_init(*fields):
                    def decorator(cls):
                        def __init__(self, *args):
                            for f, v in zip(fields, args):
                                setattr(self, f, v)
                        def __repr__(self):
                            vals = ', '.join(f'{f}={getattr(self, f)!r}' for f in fields)
                            return f'{cls.__name__}({vals})'
                        cls.__init__ = __init__
                        cls.__repr__ = __repr__
                        return cls
                    return decorator
            """),
            textwrap.dedent("""\
                @auto_init('name', 'age', 'city')
                class Person:
                    pass

                p = Person('Alice', 30, 'NYC')
                print(f"p={p}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p=Person(name='Alice', age=30, city='NYC')" in nb_runner.get_output(2)
