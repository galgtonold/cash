"""
Batch 296: Chained method calls and builder pattern interaction tests.
Tests that editing builder/fluent API patterns properly invalidates
the final built object downstream.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestBuilderPatternInteraction:
    """Test builder/fluent chain patterns with cache invalidation."""

    def test_builder_pattern_edit(self, nb_runner):
        """Editing builder calls should propagate to final built object."""
        nb_runner.create_notebook([
            (
                "class QueryBuilder:\n"
                "    def __init__(self):\n"
                "        self._parts = []\n"
                "    def select(self, fields):\n"
                "        self._parts.append(f'SELECT {fields}')\n"
                "        return self\n"
                "    def from_table(self, table):\n"
                "        self._parts.append(f'FROM {table}')\n"
                "        return self\n"
                "    def where(self, cond):\n"
                "        self._parts.append(f'WHERE {cond}')\n"
                "        return self\n"
                "    def build(self):\n"
                "        return ' '.join(self._parts)"
            ),
            "q = QueryBuilder().select('*').from_table('users').where('age > 18').build()",
            "print(f'query={q}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "query=SELECT * FROM users WHERE age > 18" in out

        nb_runner.set_cell_source(2, "q = QueryBuilder().select('name, email').from_table('employees').where('active = 1').build()")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "query=SELECT name, email FROM employees WHERE active = 1" in out

    def test_method_chain_with_intermediate_edit(self, nb_runner):
        """Editing intermediate chain steps should propagate."""
        nb_runner.create_notebook([
            (
                "class Pipeline:\n"
                "    def __init__(self, data):\n"
                "        self.data = data\n"
                "    def filter_gt(self, threshold):\n"
                "        self.data = [x for x in self.data if x > threshold]\n"
                "        return self\n"
                "    def multiply(self, factor):\n"
                "        self.data = [x * factor for x in self.data]\n"
                "        return self\n"
                "    def result(self):\n"
                "        return self.data"
            ),
            "initial = [1, 5, 3, 8, 2, 7]",
            "res = Pipeline(initial).filter_gt(3).multiply(10).result()",
            "print(f'res={sorted(res)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "res=[50, 70, 80]" in out

        nb_runner.set_cell_source(2, "initial = [10, 50, 30, 80, 20, 70]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "res=[100, 200, 300, 500, 700, 800]" in out

    def test_config_builder_edit(self, nb_runner):
        """Editing config builder steps should propagate to final config."""
        nb_runner.create_notebook([
            (
                "class ConfigBuilder:\n"
                "    def __init__(self):\n"
                "        self._config = {}\n"
                "    def set(self, key, val):\n"
                "        self._config[key] = val\n"
                "        return self\n"
                "    def build(self):\n"
                "        return dict(self._config)"
            ),
            "cfg = ConfigBuilder().set('host', 'localhost').set('port', 8080).build()",
            "info = f\"{cfg['host']}:{cfg['port']}\"",
            "print(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=localhost:8080" in out

        nb_runner.set_cell_source(2, "cfg = ConfigBuilder().set('host', 'prod.server.com').set('port', 443).build()")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=prod.server.com:443" in out
