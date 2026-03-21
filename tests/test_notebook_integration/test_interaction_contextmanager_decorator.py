"""
Interaction test: contextmanager decorator for custom context managers.
Tests @contextmanager from contextlib with yield-based resource management,
exception handling in context, and cross-cell state.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContextmanagerDecorator:
    """Test @contextmanager decorator across cells."""

    def test_contextmanager_basic(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define context manager
            "from contextlib import contextmanager\n@contextmanager\ndef track_state(name):\n    state = {'entered': True, 'name': name, 'exited': False}\n    try:\n        yield state\n    finally:\n        state['exited'] = True\nprint('track_state defined')",
            # Cell 2: use context manager
            "with track_state('test') as s:\n    s['value'] = 42\n    inside_name = s['name']\nprint(f'name={inside_name}')\nprint(f'exited={s[\"exited\"]}')\nprint(f'value={s[\"value\"]}')",
            # Cell 3: reference results
            "summary = f'{inside_name}:{s[\"value\"]}'\nprint(f'summary={summary}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "name=test" in out2
        assert "exited=True" in out2
        assert "value=42" in out2
        out3 = nb_runner.get_output(3)
        assert "summary=test:42" in out3

    def test_contextmanager_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import contextmanager\n@contextmanager\ndef counter():\n    c = [0]\n    yield c\n    c[0] += 1  # increment on exit\nprint('counter defined')",
            "with counter() as c:\n    c[0] = 10\nresult = c[0]\nprint(f'result={result}')",
            "doubled = result * 2\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(2)
        assert "doubled=22" in nb_runner.get_output(3)

        # Edit initial value
        nb_runner.set_cell_source(2, "with counter() as c:\n    c[0] = 20\nresult = c[0]\nprint(f'result={result}')")
        nb_runner.run_cells([2, 3])
        assert "result=21" in nb_runner.get_output(2)
        assert "doubled=42" in nb_runner.get_output(3)

    def test_contextmanager_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from contextlib import contextmanager\n@contextmanager\ndef tag(name):\n    result = [f'<{name}>']\n    yield result\n    result.append(f'</{name}>')\nprint('tag defined')",
            "with tag('div') as parts:\n    parts.append('content')\nhtml = ''.join(parts)\nprint(f'html={html}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "html=<div>content</div>" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "html=<div>content</div>" in nb_runner.get_output(2)
