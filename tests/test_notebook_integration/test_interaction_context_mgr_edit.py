"""Batch 270 – Context manager pattern edits.

Tests custom context managers with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContextManagerEdits:
    """Custom context manager edit patterns."""

    def test_context_manager_class_edit(self, nb_runner):
        """Edit context manager class."""
        nb_runner.create_notebook([
            "class Timer:\n    def __init__(self, label):\n        self.label = label\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass\n    def report(self):\n        return f'{self.label}: done'",
            "with Timer('task1') as t:\n    result = t.report()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = task1: done" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Timer:\n    def __init__(self, label):\n        self.label = label\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass\n    def report(self):\n        return f'[{self.label}] complete'",
        )
        nb_runner.run_all()
        assert "result = [task1] complete" in nb_runner.get_output(2)

    def test_contextmanager_decorator_edit(self, nb_runner):
        """Edit contextlib-based context manager."""
        nb_runner.create_notebook([
            "from contextlib import contextmanager\n@contextmanager\ndef managed(name):\n    yield f'resource:{name}'",
            "with managed('db') as r:\n    val = r\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = resource:db" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from contextlib import contextmanager\n@contextmanager\ndef managed(name):\n    yield f'conn:{name}:active'",
        )
        nb_runner.run_all()
        assert "val = conn:db:active" in nb_runner.get_output(2)

    def test_with_statement_usage_edit(self, nb_runner):
        """Edit the with statement usage, keep manager same."""
        nb_runner.create_notebook([
            "from contextlib import contextmanager\n@contextmanager\ndef scope(label):\n    yield label.upper()",
            "with scope('alpha') as s:\n    result = s\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ALPHA" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "with scope('beta') as s:\n    result = s\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = BETA" in nb_runner.get_output(2)
