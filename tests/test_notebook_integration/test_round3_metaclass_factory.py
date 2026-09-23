"""metaclass and class factory patterns."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


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
