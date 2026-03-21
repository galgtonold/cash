"""Batch 198 – Builder / fluent API pattern interaction tests.

Tests editing builder-style method chains and fluent interfaces.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestBuilderEdits:
    """Editing builder pattern code."""

    def test_edit_builder_step(self, nb_runner):
        """Edit one step in a builder chain."""
        nb_runner.create_notebook([
            "class QueryBuilder:\n    def __init__(self):\n        self._parts = []\n    def select(self, cols):\n        self._parts.append(f'SELECT {cols}')\n        return self\n    def where(self, cond):\n        self._parts.append(f'WHERE {cond}')\n        return self\n    def build(self):\n        return ' '.join(self._parts)",
            "q = QueryBuilder().select('*').where('id > 5').build()\nprint(f'q = {q}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "q = SELECT * WHERE id > 5" in nb_runner.get_output(2)

        # Edit the query
        nb_runner.set_cell_source(
            2,
            "q = QueryBuilder().select('name, age').where('age >= 18').build()\nprint(f'q = {q}')",
        )
        nb_runner.run_all()
        assert "q = SELECT name, age WHERE age >= 18" in nb_runner.get_output(2)

    def test_edit_builder_class(self, nb_runner):
        """Edit the builder class to add a new method."""
        nb_runner.create_notebook([
            "class HtmlBuilder:\n    def __init__(self):\n        self._html = ''\n    def tag(self, name, content):\n        self._html += f'<{name}>{content}</{name}>'\n        return self\n    def build(self):\n        return self._html",
            "result = HtmlBuilder().tag('h1', 'Title').tag('p', 'Body').build()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = <h1>Title</h1><p>Body</p>" in nb_runner.get_output(2)

        # Add wrap method
        nb_runner.set_cell_source(
            1,
            "class HtmlBuilder:\n    def __init__(self):\n        self._html = ''\n    def tag(self, name, content):\n        self._html += f'<{name}>{content}</{name}>'\n        return self\n    def wrap(self, name):\n        self._html = f'<{name}>{self._html}</{name}>'\n        return self\n    def build(self):\n        return self._html",
        )
        nb_runner.set_cell_source(
            2,
            "result = HtmlBuilder().tag('h1', 'Hi').wrap('div').build()\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = <div><h1>Hi</h1></div>" in nb_runner.get_output(2)

    def test_edit_chain_length(self, nb_runner):
        """Edit a chain by adding/removing steps."""
        nb_runner.create_notebook([
            "class Pipeline:\n    def __init__(self, val):\n        self.val = val\n    def add(self, n):\n        self.val += n\n        return self\n    def mul(self, n):\n        self.val *= n\n        return self\n    def get(self):\n        return self.val",
            "result = Pipeline(1).add(9).mul(2).get()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (1+9)*2 = 20
        assert "result = 20" in nb_runner.get_output(2)

        # Add more steps
        nb_runner.set_cell_source(
            2,
            "result = Pipeline(1).add(9).mul(2).add(5).mul(3).get()\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        # (1+9)*2=20, 20+5=25, 25*3=75
        assert "result = 75" in nb_runner.get_output(2)
