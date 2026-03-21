"""
Interaction test: recursive dict merge and nested dict access.
Tests recursive dict merging, nested key access with get(),
and cross-cell config override patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestRecursiveDictMerge:
    """Test recursive dict merge across cells."""

    def test_recursive_merge(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define merge function
            "def deep_merge(base, override):\n    result = base.copy()\n    for k, v in override.items():\n        if k in result and isinstance(result[k], dict) and isinstance(v, dict):\n            result[k] = deep_merge(result[k], v)\n        else:\n            result[k] = v\n    return result\nprint('deep_merge defined')",
            # Cell 2: merge configs
            "base = {'db': {'host': 'localhost', 'port': 5432}, 'debug': False, 'log': 'info'}\noverride = {'db': {'port': 3306, 'name': 'mydb'}, 'debug': True}\nmerged = deep_merge(base, override)\nprint(f'host={merged[\"db\"][\"host\"]}')\nprint(f'port={merged[\"db\"][\"port\"]}')\nprint(f'name={merged[\"db\"][\"name\"]}')\nprint(f'debug={merged[\"debug\"]}')\nprint(f'log={merged[\"log\"]}')",
            # Cell 3: count keys
            "total_keys = len(merged) + len(merged['db'])\nprint(f'total_keys={total_keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "host=localhost" in out2
        assert "port=3306" in out2
        assert "name=mydb" in out2
        assert "debug=True" in out2
        assert "log=info" in out2
        out3 = nb_runner.get_output(3)
        assert "total_keys=6" in out3

    def test_merge_edit(self, nb_runner):
        nb_runner.create_notebook([
            "def deep_merge(base, override):\n    result = base.copy()\n    for k, v in override.items():\n        if k in result and isinstance(result[k], dict) and isinstance(v, dict):\n            result[k] = deep_merge(result[k], v)\n        else:\n            result[k] = v\n    return result\nprint('deep_merge defined')",
            "base = {'a': 1, 'b': {'x': 10}}\nover = {'b': {'y': 20}}\nm = deep_merge(base, over)\nprint(f'result={m}')",
            "b_keys = sorted(m['b'].keys())\nprint(f'b_keys={b_keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b_keys=['x', 'y']" in nb_runner.get_output(3)

        # Edit override
        nb_runner.set_cell_source(2, "base = {'a': 1, 'b': {'x': 10}}\nover = {'b': {'y': 20, 'z': 30}, 'c': 3}\nm = deep_merge(base, over)\nprint(f'result={m}')")
        nb_runner.run_cells([2, 3])
        assert "b_keys=['x', 'y', 'z']" in nb_runner.get_output(3)

    def test_merge_cache(self, nb_runner):
        nb_runner.create_notebook([
            "def deep_merge(a, b):\n    r = a.copy()\n    for k, v in b.items():\n        if k in r and isinstance(r[k], dict) and isinstance(v, dict):\n            r[k] = deep_merge(r[k], v)\n        else:\n            r[k] = v\n    return r\nprint('defined')",
            "cfg = deep_merge({'x': 1}, {'y': 2})\nprint(f'keys={sorted(cfg.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)
