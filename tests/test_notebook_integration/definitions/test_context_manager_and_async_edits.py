"""Editing context managers and async functions."""

import pytest

pytestmark = [pytest.mark.stress]


class TestContextManagerEdits:
    """Context manager patterns with edits."""

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_context_body(self, nb_runner, tmp_path):
        """Edit code inside context manager."""
        fpath = tmp_path / "test.txt"
        fpath.write_text("hello world", encoding="utf-8")
        fpath_str = str(fpath).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"path = '{fpath_str}'",
                "with open(path) as f:\n    content = f.read()\nresult = len(content)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        # Edit to get word count instead
        nb_runner.set_cell_source(
            2,
            "with open(path) as f:\n    content = f.read()\nresult = len(content.split())\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 2" in nb_runner.get_output(2)

    # Context manager pattern edits.
    #
    # Tests custom context managers with edits.
    @pytest.mark.timeout(90)
    def test_context_manager_class_edit(self, nb_runner):
        """Edit context manager class."""
        nb_runner.create_notebook(
            [
                "class Timer:\n    def __init__(self, label):\n        self.label = label\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass\n    def report(self):\n        return f'{self.label}: done'",
                "with Timer('task1') as t:\n    result = t.report()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = task1: done" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Timer:\n    def __init__(self, label):\n        self.label = label\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass\n    def report(self):\n        return f'[{self.label}] complete'",
        )
        nb_runner.run_all()
        assert "result = [task1] complete" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_contextmanager_decorator_edit(self, nb_runner):
        """Edit contextlib-based context manager."""
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef managed(name):\n    yield f'resource:{name}'",
                "with managed('db') as r:\n    val = r\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = resource:db" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from contextlib import contextmanager\n@contextmanager\ndef managed(name):\n    yield f'conn:{name}:active'",
        )
        nb_runner.run_all()
        assert "val = conn:db:active" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_with_statement_usage_edit(self, nb_runner):
        """Edit the with statement usage, keep manager same."""
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef scope(label):\n    yield label.upper()",
                "with scope('alpha') as s:\n    result = s\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ALPHA" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "with scope('beta') as s:\n    result = s\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = BETA" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestAsyncPatterns:
    """Async/await edit propagation."""

    def test_async_function_edit(self, nb_runner):
        """Edit async function, await result changes."""
        nb_runner.create_notebook(
            [
                "import asyncio\nasync def compute(x):\n    return x * 2",
                "result = await compute(21)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "import asyncio\nasync def compute(x):\n    return x ** 2",
        )
        nb_runner.run_all()
        assert "result = 441" in nb_runner.get_output(2)

    def test_async_gather_edit(self, nb_runner):
        """Edit async tasks gathered together."""
        nb_runner.create_notebook(
            [
                "import asyncio\nasync def task(n):\n    return n + 1",
                "result = list(await asyncio.gather(task(1), task(2), task(3)))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "import asyncio\nasync def task(n):\n    return n * 10",
        )
        nb_runner.run_all()
        assert "result = [10, 20, 30]" in nb_runner.get_output(2)
