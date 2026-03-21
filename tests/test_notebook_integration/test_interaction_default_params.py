"""Batch 269 – Default parameter and keyword argument edit patterns.

Tests functions with default/keyword args, editing defaults.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDefaultParamEdits:
    """Default parameter edit propagation."""

    def test_default_param_edit(self, nb_runner):
        """Edit function default parameter."""
        nb_runner.create_notebook([
            "def greet(name, greeting='Hello'):\n    return f'{greeting}, {name}!'",
            "msg = greet('Alice')\nprint(f'msg = {msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = Hello, Alice!" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def greet(name, greeting='Hi'):\n    return f'{greeting}, {name}!'",
        )
        nb_runner.run_all()
        assert "msg = Hi, Alice!" in nb_runner.get_output(2)

    def test_kwargs_dict_edit(self, nb_runner):
        """Edit kwargs dict passed to function."""
        nb_runner.create_notebook([
            "def build_config(**kwargs):\n    return {k: v for k, v in sorted(kwargs.items())}",
            "opts = {'color': 'blue', 'size': 10}",
            "cfg = build_config(**opts)\nprint(f'cfg = {cfg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'color': 'blue'" in nb_runner.get_output(3)
        assert "'size': 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "opts = {'color': 'red', 'size': 20, 'weight': 5}")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'color': 'red'" in out
        assert "'weight': 5" in out

    def test_mixed_args_edit(self, nb_runner):
        """Edit function with mixed positional and keyword args."""
        nb_runner.create_notebook([
            "def format_record(name, age, role='member'):\n    return f'{name} ({age}) - {role}'",
            "record = format_record('Alice', 30)\nprint(f'record = {record}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "record = Alice (30) - member" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def format_record(name, age, role='admin'):\n    return f'{name} ({age}) - {role}'",
        )
        nb_runner.run_all()
        assert "record = Alice (30) - admin" in nb_runner.get_output(2)
