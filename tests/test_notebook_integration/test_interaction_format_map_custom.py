"""
Interaction test: string formatting with format_map and template patterns.
Tests str.format_map, custom Mapping classes for format,
and cross-cell string formatting pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFormatMapCustom:
    """Test str.format_map with custom mappings across cells."""

    def test_format_map_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: format_map with defaultdict
            "from collections import defaultdict\ndata = defaultdict(lambda: 'N/A', name='Alice', age=30)\nresult = '{name} is {age}, lives in {city}'.format_map(data)\nprint(f'result={result}')",
            # Cell 2: format with dict subclass
            "class SafeDict(dict):\n    def __missing__(self, key):\n        return f'<{key}>'\n\nsd = SafeDict(x=10, y=20)\nformatted = '{x} + {y} = {z}'.format_map(sd)\nprint(f'formatted={formatted}')",
            # Cell 3: complex formatting
            "template = 'Name: {name}, Score: {score:.1f}, Grade: {grade}'\nstudents = [\n    {'name': 'Alice', 'score': 95.5, 'grade': 'A'},\n    {'name': 'Bob', 'score': 82.3, 'grade': 'B'},\n]\nfor s in students:\n    print(template.format_map(s))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "result=Alice is 30, lives in N/A" in out1
        out2 = nb_runner.get_output(2)
        assert "formatted=10 + 20 = <z>" in out2
        out3 = nb_runner.get_output(3)
        assert "Name: Alice, Score: 95.5, Grade: A" in out3
        assert "Name: Bob, Score: 82.3, Grade: B" in out3

    def test_format_map_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = {'product': 'Widget', 'price': 9.99}\nmsg = 'Buy {product} for ${price:.2f}'.format_map(data)\nprint(f'msg={msg}')",
            "upper = msg.upper()\nprint(f'upper={upper}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Buy Widget for $9.99" in nb_runner.get_output(1)

        # Edit data
        nb_runner.set_cell_source(1, "data = {'product': 'Gadget', 'price': 19.99}\nmsg = 'Buy {product} for ${price:.2f}'.format_map(data)\nprint(f'msg={msg}')")
        nb_runner.run_cells([1, 2])
        assert "msg=Buy Gadget for $19.99" in nb_runner.get_output(1)

    def test_format_map_cache(self, nb_runner):
        nb_runner.create_notebook([
            "info = {'name': 'test', 'version': '1.0'}\nheader = '{name} v{version}'.format_map(info)\nprint(f'header={header}')",
            "length = len(header)\nprint(f'length={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header=test v1.0" in nb_runner.get_output(1)
        # "test v1.0" = 9 chars
        assert "length=9" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=9" in nb_runner.get_output(2)
