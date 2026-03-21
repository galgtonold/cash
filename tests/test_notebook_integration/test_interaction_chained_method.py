"""Batch 212 – Method chaining interaction tests.

Tests editing cells that use method chaining patterns
(string chains, list chains, custom fluent APIs).
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMethodChainingEdits:
    """Editing method chaining patterns."""

    def test_edit_string_method_chain(self, nb_runner):
        """Edit input to a string method chain."""
        nb_runner.create_notebook([
            "text = '  Hello, World!  '",
            "result = text.strip().lower().replace(',', '')\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hello world!" in nb_runner.get_output(2)

        # Change input text
        nb_runner.set_cell_source(1, "text = '  GOODBYE, WORLD!  '")
        nb_runner.run_all()
        assert "result = goodbye world!" in nb_runner.get_output(2)

    def test_edit_list_method_chain(self, nb_runner):
        """Edit list operations chain."""
        nb_runner.create_notebook([
            "data = [3, 1, 4, 1, 5, 9, 2, 6]",
            "result = sorted(set(data))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5, 6, 9]" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [7, 7, 3, 3, 1, 1]")
        nb_runner.run_all()
        assert "result = [1, 3, 7]" in nb_runner.get_output(2)

    def test_edit_chain_middle_step(self, nb_runner):
        """Edit the middle step of a processing chain."""
        nb_runner.create_notebook([
            "words = ['hello', 'world', 'foo', 'bar', 'baz']",
            "result = [w.upper() for w in words if len(w) > 3]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['HELLO', 'WORLD']" in nb_runner.get_output(2)

        # Change filter threshold
        nb_runner.set_cell_source(2, "result = [w.upper() for w in words if len(w) >= 3]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "FOO" in nb_runner.get_output(2)
        assert "BAR" in nb_runner.get_output(2)
        assert "BAZ" in nb_runner.get_output(2)

    def test_edit_dict_chain(self, nb_runner):
        """Edit dict used in chained transformations."""
        nb_runner.create_notebook([
            "inventory = {'apple': 5, 'banana': 2, 'cherry': 8, 'date': 1}",
            "available = {k: v for k, v in inventory.items() if v > 2}\nnames = sorted(available.keys())\nprint(f'names = {names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['apple', 'cherry']" in nb_runner.get_output(2)

        # Update inventory
        nb_runner.set_cell_source(1, "inventory = {'apple': 1, 'banana': 10, 'cherry': 8, 'date': 5}")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "banana" in out
        assert "cherry" in out
        assert "date" in out
