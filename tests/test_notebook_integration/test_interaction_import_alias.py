"""Batch 188 – Import alias and module-level function interaction tests.

Tests editing import aliases, switching between import styles,
and using module-level functions with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.modules, pytest.mark.timeout(90)]


class TestImportAliasEdits:
    """Editing import alias patterns."""

    def test_edit_import_alias(self, nb_runner):
        """Change an import alias and verify downstream uses."""
        nb_runner.create_notebook([
            "import math as m  # alias v1",
            "result = m.sqrt(144)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12.0" in nb_runner.get_output(2)

        # Change to use different function
        nb_runner.set_cell_source(
            2, "result = m.factorial(5)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

    def test_switch_import_style(self, nb_runner):
        """Switch between import styles."""
        nb_runner.create_notebook([
            "from os.path import join  # from import style",
            "result = join('a', 'b', 'c')\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a" in out and "b" in out and "c" in out

        # Switch to module import
        nb_runner.set_cell_source(1, "import os.path  # module import style")
        nb_runner.set_cell_source(
            2, "result = os.path.join('x', 'y', 'z')\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "x" in out2 and "y" in out2 and "z" in out2


class TestModuleFunctionEdits:
    """Editing usage of module-level functions."""

    def test_edit_module_function_args(self, nb_runner):
        """Edit arguments to module functions."""
        nb_runner.create_notebook([
            "import json",
            "data = json.dumps({'a': 1})\nprint(f'data = {data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert 'data = {"a": 1}' in nb_runner.get_output(2)

        # Change to pretty print
        nb_runner.set_cell_source(
            2,
            "data = json.dumps({'a': 1, 'b': 2}, indent=2)\nprint(f'len = {len(data)}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len = " in out

    def test_edit_collections_usage(self, nb_runner):
        """Edit usage of collections module functions."""
        nb_runner.create_notebook([
            "from collections import Counter",
            "c = Counter('aabbcc')\nprint(f'c = {dict(c)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 2" in nb_runner.get_output(2)

        # Change input
        nb_runner.set_cell_source(
            2, "c = Counter('aaabbb')\nprint(f'c = {dict(c)}')"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 3" in out
        assert "'b': 3" in out

    def test_edit_itertools_usage(self, nb_runner):
        """Edit itertools pipeline."""
        nb_runner.create_notebook([
            "from itertools import chain, repeat",
            "result = list(chain([1, 2], [3, 4]))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to repeat
        nb_runner.set_cell_source(
            2, "result = list(repeat(42, 3))\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = [42, 42, 42]" in nb_runner.get_output(2)
