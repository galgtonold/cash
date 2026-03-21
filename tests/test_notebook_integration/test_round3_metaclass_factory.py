"""Batch 89 – metaclass and class factory patterns."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestMetaclass:
    """Metaclass patterns."""

    def test_basic_metaclass(self, nb_runner):
        """Simple metaclass that adds a registry."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class RegistryMeta(type):
                    _registry = {}
                    def __new__(mcs, name, bases, namespace):
                        cls = super().__new__(mcs, name, bases, namespace)
                        if name != 'Base':
                            mcs._registry[name] = cls
                        return cls

                class Base(metaclass=RegistryMeta):
                    pass

                class Dog(Base):
                    sound = 'woof'

                class Cat(Base):
                    sound = 'meow'

                class Bird(Base):
                    sound = 'tweet'

                registered = sorted(RegistryMeta._registry.keys())
            """),
            "print(f'registered={registered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Bird" in out
        assert "Cat" in out
        assert "Dog" in out

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

                class Database(metaclass=SingletonMeta):
                    def __init__(self):
                        self.connection = 'connected'

                db1 = Database()
                db2 = Database()
                same = db1 is db2
            """),
            "print(f'same={same} conn={db1.connection}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "same=True" in out
        assert "conn=connected" in out

    def test_metaclass_validation(self, nb_runner):
        """Metaclass that validates class attributes."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "valid=True" in out
        assert "host=localhost" in out
        assert "port=8080" in out


class TestClassFactory:
    """Dynamic class creation patterns."""

    def test_type_factory(self, nb_runner):
        """Create classes dynamically with type()."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def make_model(name, fields):
                    def init(self, **kwargs):
                        for f in fields:
                            setattr(self, f, kwargs.get(f))
                    def repr_fn(self):
                        vals = ', '.join(f'{f}={getattr(self, f)!r}' for f in fields)
                        return f'{type(self).__name__}({vals})'
                    return type(name, (), {'__init__': init, '__repr__': repr_fn, '_fields': fields})

                User = make_model('User', ['name', 'age'])
                Product = make_model('Product', ['title', 'price'])
                u = User(name='Alice', age=30)
                p = Product(title='Widget', price=9.99)
            """),
            "print(f'user={u}')\nprint(f'product={p}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Alice" in out
        assert "Widget" in out
        assert "9.99" in out

    def test_factory_propagation(self, nb_runner):
        """Class factory with upstream field change propagation."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x" in nb_runner.get_output(3)
        assert "y" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "fields = ['x', 'y', 'z']")
        nb_runner.run_cells([1, 2, 3])
        out = nb_runner.get_output(3)
        assert "z" in out
