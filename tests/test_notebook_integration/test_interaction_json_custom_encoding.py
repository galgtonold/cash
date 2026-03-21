"""
Interaction test: json loads and dumps with custom encoding.
Tests json serialization with custom defaults, indent,
sort_keys, and cross-cell data transformation pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestJsonCustomEncoding:
    """Test json loads/dumps with custom encoding across cells."""

    def test_json_custom(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: custom json serialization
            "import json\nfrom datetime import datetime, date\n\nclass DateEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, (datetime, date)):\n            return obj.isoformat()\n        return super().default(obj)\n\ndata = {'name': 'Alice', 'created': date(2024, 1, 15), 'scores': [95, 87, 92]}\njson_str = json.dumps(data, cls=DateEncoder, sort_keys=True)\nprint(f'json={json_str}')",
            # Cell 2: parse back
            "parsed = json.loads(json_str)\nprint(f'name={parsed[\"name\"]}')\nprint(f'created={parsed[\"created\"]}')\nprint(f'scores_sum={sum(parsed[\"scores\"])}')",
            # Cell 3: pretty print
            "pretty = json.dumps(parsed, indent=2, sort_keys=True)\nline_count = len(pretty.split('\\n'))\nprint(f'lines={line_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "2024-01-15" in out1
        assert "Alice" in out1
        out2 = nb_runner.get_output(2)
        assert "name=Alice" in out2
        assert "scores_sum=274" in out2
        out3 = nb_runner.get_output(3)
        assert "lines=" in out3

    def test_json_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import json\nconfig = {'debug': True, 'port': 8080}\njson_str = json.dumps(config, sort_keys=True)\nprint(f'json={json_str}')",
            "parsed = json.loads(json_str)\nport = parsed['port']\nprint(f'port={port}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "port=8080" in nb_runner.get_output(2)

        # Edit config
        nb_runner.set_cell_source(1, "import json\nconfig = {'debug': False, 'port': 9090, 'host': 'localhost'}\njson_str = json.dumps(config, sort_keys=True)\nprint(f'json={json_str}')")
        nb_runner.run_cells([1, 2])
        assert "port=9090" in nb_runner.get_output(2)

    def test_json_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import json\nitems = [{'id': 1, 'val': 'a'}, {'id': 2, 'val': 'b'}]\njson_str = json.dumps(items)\nprint(f'length={len(json_str)}')",
            "back = json.loads(json_str)\ncount = len(back)\nprint(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
