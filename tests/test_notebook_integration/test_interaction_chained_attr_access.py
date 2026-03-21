"""Batch 350: chained attribute access and method chains with edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainedAttributeAccess:
    def test_chained_methods(self, nb_runner):
        nb_runner.create_notebook([
            "class Builder:\n    def __init__(self):\n        self.parts = []\n    def add(self, part):\n        self.parts.append(part)\n        return self\n    def build(self):\n        return '-'.join(self.parts)",
            "result = Builder().add('a').add('b').add('c').build()\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=a-b-c" in nb_runner.get_output(2)

    def test_chained_edit_class(self, nb_runner):
        nb_runner.create_notebook([
            "class Query:\n    def __init__(self):\n        self.filters = []\n    def where(self, cond):\n        self.filters.append(cond)\n        return self\n    def execute(self):\n        return f'filters={self.filters}'",
            "q = Query().where('x>5').where('y<10').execute()\nprint(f'q={q}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filters=['x>5', 'y<10']" in nb_runner.get_output(2)
        # Edit class to add limit
        nb_runner.set_cell_source(1, "class Query:\n    def __init__(self):\n        self.filters = []\n        self.limit_val = None\n    def where(self, cond):\n        self.filters.append(cond)\n        return self\n    def limit(self, n):\n        self.limit_val = n\n        return self\n    def execute(self):\n        return f'filters={self.filters} limit={self.limit_val}'")
        nb_runner.set_cell_source(2, "q = Query().where('x>5').limit(100).execute()\nprint(f'q={q}')")
        nb_runner.run_all()
        assert "filters=['x>5'] limit=100" in nb_runner.get_output(2)

    def test_nested_attribute(self, nb_runner):
        nb_runner.create_notebook([
            "class Inner:\n    def __init__(self, v):\n        self.value = v\nclass Outer:\n    def __init__(self, v):\n        self.inner = Inner(v)",
            "o = Outer(42)\nresult = o.inner.value\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=42" in nb_runner.get_output(2)
