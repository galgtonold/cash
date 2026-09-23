"""Property & descriptor patterns — cash caching with properties and descriptors."""

import textwrap

import pytest


@pytest.mark.stress
class TestDescriptorPatterns:
    """Test descriptor protocol patterns across cells."""

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
