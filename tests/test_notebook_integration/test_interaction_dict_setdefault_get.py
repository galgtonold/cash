"""Batch 461: dict setdefault and get with defaults."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictSetdefaultGet:
    def test_setdefault(self, nb_runner):
        nb_runner.create_notebook([
            "d = {'a': 1}",
            "d.setdefault('a', 99)\nd.setdefault('b', 42)\nprint(f'a={d[\"a\"]} b={d[\"b\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1" in nb_runner.get_output(2)
        assert "b=42" in nb_runner.get_output(2)

    def test_get_default(self, nb_runner):
        nb_runner.create_notebook([
            "config = {'debug': True, 'port': 8080}",
            "debug = config.get('debug', False)\nhost = config.get('host', 'localhost')\nport = config.get('port', 3000)\nprint(f'debug={debug} host={host} port={port}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "debug=True" in nb_runner.get_output(2)
        assert "host=localhost" in nb_runner.get_output(2)
        assert "port=8080" in nb_runner.get_output(2)

    def test_setdefault_edit(self, nb_runner):
        nb_runner.create_notebook([
            "groups = {}",
            "for item in ['a', 'b', 'a', 'c', 'b', 'a']:\n    groups.setdefault(item, 0)\n    groups[item] += 1\nprint(f'groups={groups}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 3" in nb_runner.get_output(2)
        assert "'b': 2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "groups = {}")
        nb_runner.set_cell_source(2, "for item in ['x', 'x', 'y']:\n    groups.setdefault(item, 0)\n    groups[item] += 1\nprint(f'groups={groups}')")
        nb_runner.run_all()
        assert "'x': 2" in nb_runner.get_output(2)
        assert "'y': 1" in nb_runner.get_output(2)
