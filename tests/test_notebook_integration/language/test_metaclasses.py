"""Metaclasses and class factories across cells."""

import textwrap

import pytest


@pytest.mark.stress
class TestMetaclassBasics:
    """Test basic metaclass usage."""

    def test_metaclass_change_propagates(self, nb_runner):
        """Changing metaclass-created class propagates."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "version=1.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            class API(metaclass=Versioned):
                VERSION = '2.0'
            print(f"version={API.version}")
        """),
        )
        nb_runner.run_all()
        assert "version=2.0" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.integration
class TestMetaclass:
    """Metaclass patterns."""

    def test_metaclass_validation(self, nb_runner):
        """Metaclass that validates class attributes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class ValidatedMeta(type):
                    def __new__(mcs, name, bases, namespace):
                        required = namespace.get('_required_attrs', [])
                        missing = [a for a in required if a not in namespace]
                        if missing:
                            raise TypeError(f"{name} missing: {missing}")
                        cls = super().__new__(mcs, name, bases, namespace)
                        return cls

                class Config(metaclass=ValidatedMeta):
                    _required_attrs = ['host', 'port']
                    host = 'localhost'
                    port = 8080
                    debug = True

                valid = True
                config_host = Config.host
                config_port = Config.port
            """),
                "print(f'valid={valid} host={config_host} port={config_port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "valid=True" in out
        assert "host=localhost" in out
        assert "port=8080" in out


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMetaclassInteraction:
    """Test metaclass patterns with cache invalidation."""

    def test_registry_metaclass_edit(self, nb_runner):
        """Editing a class with a registry metaclass should propagate.

        Note: Registry._registry is mutated as a side-effect of class creation,
        so downstream cells must reference the classes directly (Alpha, Beta)
        to make the dependency explicit for lineage tracking.
        """
        nb_runner.create_notebook(
            [
                (
                    "class Registry(type):\n"
                    "    _registry = {}\n"
                    "    def __new__(mcs, name, bases, namespace):\n"
                    "        cls = super().__new__(mcs, name, bases, namespace)\n"
                    "        if name != 'Base':\n"
                    "            mcs._registry[name] = cls\n"
                    "        return cls\n"
                    "\n"
                    "class Base(metaclass=Registry):\n"
                    "    pass"
                ),
                "class Alpha(Base):\n    value = 1\n\nclass Beta(Base):\n    value = 2",
                "total = Alpha.value + Beta.value",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=3" in out

        nb_runner.set_cell_source(2, "class Alpha(Base):\n    value = 10\n\nclass Beta(Base):\n    value = 20")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=30" in out

    def test_singleton_metaclass_edit(self, nb_runner):
        """Editing a singleton class should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Singleton(type):\n"
                    "    _instances = {}\n"
                    "    def __call__(cls, *args, **kwargs):\n"
                    "        if cls not in cls._instances:\n"
                    "            cls._instances[cls] = super().__call__(*args, **kwargs)\n"
                    "        return cls._instances[cls]"
                ),
                (
                    "class AppConfig(metaclass=Singleton):\n"
                    "    def __init__(self, name='default'):\n"
                    "        self.name = name"
                ),
                "c1 = AppConfig('myapp')\nc2 = AppConfig('other')\nsame = c1 is c2",
                "print(f'name={c1.name},same={same}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "same=True" in out
        assert "name=myapp" in out

        # Edit to clear singleton cache
        nb_runner.set_cell_source(
            3, "Singleton._instances.clear()\nc1 = AppConfig('v2app')\nc2 = AppConfig('other')\nsame = c1 is c2"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "name=v2app" in out
        assert "same=True" in out


@pytest.mark.stress
class TestInitSubclass:
    """Test __init_subclass__ patterns."""

    def test_init_subclass_auto_register(self, nb_runner):
        """__init_subclass__ for auto-registration."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "commands=['create', 'list']" in nb_runner.get_output(3)
        assert "results=['creating', 'listing']" in nb_runner.get_output(3)


@pytest.mark.stress
class TestClassDecorators:
    """Test class-level decorators."""

    def test_dataclass_like_decorator(self, nb_runner):
        """Decorator that adds methods like dataclass."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p=Person(name='Alice', age=30, city='NYC')" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.integration
class TestClassFactory:
    """Dynamic class creation patterns."""

    def test_factory_propagation(self, nb_runner):
        """Class factory with upstream field change propagation."""
        nb_runner.create_notebook(
            [
                "fields = ['x', 'y']",
                textwrap.dedent("""\
                def make_point(field_list):
                    def init(self, **kwargs):
                        for f in field_list:
                            setattr(self, f, kwargs.get(f, 0))
                    def to_dict(self):
                        return {f: getattr(self, f) for f in field_list}
                    return type('Point', (), {'__init__': init, 'to_dict': to_dict})

                Point = make_point(fields)
                p = Point(x=1, y=2)
                d = p.to_dict()
            """),
                "print(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x" in nb_runner.get_output(3)
        assert "y" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "fields = ['x', 'y', 'z']")
        nb_runner.run_cells([1, 2, 3])
        out = nb_runner.get_output(3)
        assert "z" in out
