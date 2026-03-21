"""Batch 493: json dumps loads with custom encoder."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestJsonDumpsLoadsCustom:
    def test_json_roundtrip(self, nb_runner):
        nb_runner.create_notebook([
            "import json",
            "data = {'name': 'Alice', 'scores': [95, 87, 92], 'meta': {'grade': 'A'}}\nencoded = json.dumps(data, sort_keys=True)\ndecoded = json.loads(encoded)\nprint(f'encoded_type={type(encoded).__name__}')\nprint(f'name={decoded[\"name\"]} scores_sum={sum(decoded[\"scores\"])}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "encoded_type=str" in out
        assert "name=Alice" in out
        assert "scores_sum=274" in out

    def test_json_indent(self, nb_runner):
        nb_runner.create_notebook([
            "import json",
            "data = {'a': 1, 'b': [2, 3]}\npretty = json.dumps(data, indent=2)\nlines = pretty.strip().split('\\n')\nprint(f'lines={len(lines)} has_indent={\"  \" in pretty}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "has_indent=True" in out

    def test_json_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import json",
            "d = {'x': 1}\ns = json.dumps(d)\nprint(f's={s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert 's={"x": 1}' in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = {'x': 1, 'y': 2}\ns = json.dumps(d)\nprint(f's={s}')")
        nb_runner.run_all()
        assert '"y": 2' in nb_runner.get_output(2)
