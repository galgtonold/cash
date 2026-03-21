"""Batch 424: defaultdict with lambda and complex nesting."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDefaultdictLambdaNesting:
    def test_defaultdict_lambda(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import defaultdict\ndd = defaultdict(lambda: 'unknown')",
            "dd['a'] = 'apple'\nr1 = dd['a']\nr2 = dd['b']\nprint(f'r1={r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=apple" in nb_runner.get_output(2)
        assert "r2=unknown" in nb_runner.get_output(2)

    def test_nested_defaultdict(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import defaultdict\ndef nested(): return defaultdict(int)\ndd = defaultdict(nested)",
            "dd['math']['hw1'] = 90\ndd['math']['hw2'] = 85\ndd['eng']['hw1'] = 92\nresult = dict(dd['math'])\nprint(f'math={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "math={'hw1': 90, 'hw2': 85}" in nb_runner.get_output(2)

    def test_defaultdict_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import defaultdict\ndata = [('a', 1), ('b', 2), ('a', 3)]",
            "dd = defaultdict(list)\nfor k, v in data:\n    dd[k].append(v)\nresult = dict(dd)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': [1, 3]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import defaultdict\ndata = [('x', 10), ('y', 20), ('x', 30), ('y', 40)]")
        nb_runner.run_all()
        assert "'x': [10, 30]" in nb_runner.get_output(2)
        assert "'y': [20, 40]" in nb_runner.get_output(2)
