"""Batch 182 – Context manager pattern interaction tests.

Tests editing context manager definitions, with-statements,
and resource management patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestContextManagerEdits:
    """Editing context manager patterns."""

    def test_edit_context_manager_body(self, nb_runner):
        """Edit the body of a with statement."""
        nb_runner.create_notebook([
            "from contextlib import contextmanager",
            "@contextmanager\ndef tag(name):\n    print(f'<{name}>', end='')\n    yield\n    print(f'</{name}>', end='')",
            "import io, sys\nbuf = io.StringIO()\nold = sys.stdout\nsys.stdout = buf\nwith tag('div'):\n    print('hello', end='')\nsys.stdout = old\nresult = buf.getvalue()\nprint(repr(result))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out3 = nb_runner.get_output(3)
        assert "<div>hello</div>" in out3

        # Change tag name
        nb_runner.set_cell_source(
            3,
            "import io, sys\nbuf = io.StringIO()\nold = sys.stdout\nsys.stdout = buf\nwith tag('span'):\n    print('world', end='')\nsys.stdout = old\nresult = buf.getvalue()\nprint(repr(result))",
        )
        nb_runner.run_all()
        out3b = nb_runner.get_output(3)
        assert "<span>world</span>" in out3b

    def test_edit_cm_class(self, nb_runner):
        """Edit a class-based context manager."""
        nb_runner.create_notebook([
            "class Timer:\n    def __enter__(self):\n        self.msg = 'started'\n        return self\n    def __exit__(self, *args):\n        self.msg = 'stopped'",
            "with Timer() as t:\n    pass\nprint(f'msg = {t.msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = stopped" in nb_runner.get_output(2)

        # Change exit message
        nb_runner.set_cell_source(
            1,
            "class Timer:\n    def __enter__(self):\n        self.msg = 'started'\n        return self\n    def __exit__(self, *args):\n        self.msg = 'finished'",
        )
        nb_runner.run_all()
        assert "msg = finished" in nb_runner.get_output(2)


class TestResourcePatterns:
    """Resource management patterns."""

    def test_temp_file_context(self, nb_runner, tmp_path):
        """Edit temp file usage in a context manager."""
        fpath = str(tmp_path / "ctx_test.txt").replace("\\", "/")
        nb_runner.create_notebook([
            f"path = '{fpath}'",
            "with open(path, 'w') as f:\n    f.write('version1')",
            "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "content = version1" in nb_runner.get_output(3)

        # Edit write content
        nb_runner.set_cell_source(
            2, "with open(path, 'w') as f:\n    f.write('version2')"
        )
        nb_runner.run_all()
        assert "content = version2" in nb_runner.get_output(3)
