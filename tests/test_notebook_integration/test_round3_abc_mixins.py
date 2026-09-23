"""Abstract base classes & mixins — cash caching with ABC patterns."""

import textwrap

import pytest


@pytest.mark.stress
class TestMixinPatterns:
    """Test mixin patterns across cells."""

    def test_mixin_composition(self, nb_runner):
        """Multiple mixins composed into a class."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid=True" in nb_runner.get_output(1)
        assert '"name": "Alice"' in nb_runner.get_output(1)
        assert "u2_valid=False" in nb_runner.get_output(2)

    def test_mixin_change_propagation(self, nb_runner):
        """Mixin behavior change propagates downstream."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary=1, 2, 3, 4, 5" in nb_runner.get_output(2)

        # Change separator
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
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
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "summary=1 | 2 | 3 | 4 | 5" in nb_runner.get_output(2)
