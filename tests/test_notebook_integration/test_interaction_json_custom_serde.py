"""Batch 358: json serialization/deserialization with custom objects."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestJsonCustomSerde:
    def test_json_custom_encoder(self, nb_runner):
        nb_runner.create_notebook([
            "import json\nfrom datetime import date\nclass DateEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, date):\n            return obj.isoformat()\n        return super().default(obj)",
            "data = {'name': 'event', 'date': date(2024, 1, 15)}\nresult = json.dumps(data, cls=DateEncoder)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert '"date": "2024-01-15"' in nb_runner.get_output(2)

    def test_json_edit_data(self, nb_runner):
        nb_runner.create_notebook([
            "import json\ndata = {'a': 1, 'b': [2, 3]}",
            "encoded = json.dumps(data, sort_keys=True)\ndecoded = json.loads(encoded)\nprint(f'encoded={encoded}')\nprint(f'match={data == decoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert 'encoded={"a": 1, "b": [2, 3]}' in out
        assert "match=True" in out
        # Edit data
        nb_runner.set_cell_source(1, "import json\ndata = {'x': [10, 20], 'y': 'hello'}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert '"x": [10, 20]' in out2
        assert "match=True" in out2

    def test_json_nested(self, nb_runner):
        nb_runner.create_notebook([
            "import json\nnested = {'level1': {'level2': {'value': 42}}}",
            "s = json.dumps(nested)\nback = json.loads(s)\nval = back['level1']['level2']['value']\nprint(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=42" in nb_runner.get_output(2)
