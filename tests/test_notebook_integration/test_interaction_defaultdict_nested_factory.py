"""Batch 516: defaultdict nested and lambda factories."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDefaultdictNestedFactory:
    def test_nested_defaultdict(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "dd = defaultdict(lambda: defaultdict(int))\ndd['fruits']['apple'] += 3\ndd['fruits']['banana'] += 2\ndd['vegs']['carrot'] += 1\nprint(f'apple={dd[\"fruits\"][\"apple\"]} banana={dd[\"fruits\"][\"banana\"]} carrot={dd[\"vegs\"][\"carrot\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple=3" in out
        assert "banana=2" in out
        assert "carrot=1" in out

    def test_defaultdict_list(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "groups = defaultdict(list)\nfor name, dept in [('Alice','Eng'), ('Bob','Sales'), ('Carol','Eng'), ('Dave','Sales')]:\n    groups[dept].append(name)\nprint(f'eng={sorted(groups[\"Eng\"])} sales={sorted(groups[\"Sales\"])}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eng=['Alice', 'Carol']" in out
        assert "sales=['Bob', 'Dave']" in out

    def test_defaultdict_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "d = defaultdict(int)\nfor c in 'hello': d[c] += 1\nprint(f'l={d[\"l\"]} o={d[\"o\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "l=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = defaultdict(int)\nfor c in 'mississippi': d[c] += 1\nprint(f's={d[\"s\"]} p={d[\"p\"]}')")
        nb_runner.run_all()
        assert "s=4" in nb_runner.get_output(2)
        assert "p=2" in nb_runner.get_output(2)
