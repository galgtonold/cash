"""Complex inheritance & MRO patterns — diamond, mixin, super() chains."""

import textwrap

import pytest


@pytest.mark.stress
class TestDiamondInheritance:
    """Test diamond inheritance and MRO."""

    def test_diamond_change_base(self, nb_runner):
        """Changing base class propagates through diamond."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)

        # Change base
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Base:
                value = 99
            class Left(Base): pass
            class Right(Base): pass
            class Diamond(Left, Right): pass
        """),
        )
        nb_runner.run_all()
        assert "result=99" in nb_runner.get_output(2)


@pytest.mark.stress
class TestMixinPatterns:
    """Test mixin class patterns."""

    def test_multiple_mixins(self, nb_runner):
        """Multiple mixins providing different features."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert '"name": "Alice"' in out
        assert "valid=True" in out

    def test_mixin_evolution(self, nb_runner):
        """Evolving mixin adds new method."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Object with" in out

        # Evolve mixin
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class PrintMixin:
                def describe(self):
                    attrs = ', '.join(f'{k}={v}' for k, v in sorted(self.__dict__.items()) if not k.startswith('_cash'))
                    return f"Object({attrs})"
        """),
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "name=Widget" in out2
        assert "price=9.99" in out2
