"""Batch 425: json serialization with custom encoders."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestJsonCustomEncoders:
    def test_json_dumps_loads(self, nb_runner):
        nb_runner.create_notebook([
            "import json\ndata = {'name': 'Alice', 'scores': [95, 87, 92]}",
            "s = json.dumps(data, sort_keys=True)\nloaded = json.loads(s)\nprint(f'name={loaded[\"name\"]} scores={loaded[\"scores\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=Alice" in nb_runner.get_output(2)
        assert "scores=[95, 87, 92]" in nb_runner.get_output(2)

    def test_json_indent(self, nb_runner):
        nb_runner.create_notebook([
            "import json\nobj = {'a': 1, 'b': [2, 3]}",
            "pretty = json.dumps(obj, indent=2)\nline_count = len(pretty.split('\\n'))\nprint(f'lines={line_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        count = int(nb_runner.get_output(2).split("lines=")[1].strip())
        assert count >= 5

    def test_json_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import json\nrecord = {'x': 10, 'y': 20}",
            "s = json.dumps(record)\nsize = len(s)\nprint(f'size={size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        first_size = int(nb_runner.get_output(2).split("size=")[1].strip())
        nb_runner.set_cell_source(1, "import json\nrecord = {'x': 10, 'y': 20, 'z': 30, 'w': 40}")
        nb_runner.run_all()
        second_size = int(nb_runner.get_output(2).split("size=")[1].strip())
        assert second_size > first_size
