"""Batch 517: context manager class enter exit pattern."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContextManagerClassPattern:
    def test_custom_context_manager(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Timer:\n    def __init__(self, name): self.name = name\n    def __enter__(self):\n        self.log = [f'enter:{self.name}']\n        return self\n    def __exit__(self, *args):\n        self.log.append(f'exit:{self.name}')\n        return False\nwith Timer('test') as t:\n    t.log.append('body')\nprint(f'log={t.log}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=['enter:test', 'body', 'exit:test']" in nb_runner.get_output(2)

    def test_nested_context(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Scope:\n    instances = []\n    def __init__(self, name): self.name = name\n    def __enter__(self):\n        Scope.instances.append(self.name)\n        return self\n    def __exit__(self, *a):\n        Scope.instances.pop()\nwith Scope('outer') as o:\n    with Scope('inner') as i:\n        snapshot = list(Scope.instances)\nprint(f'snapshot={snapshot} after={Scope.instances}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "snapshot=['outer', 'inner']" in out
        assert "after=[]" in out

    def test_context_edit(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Ctx:\n    def __init__(self, v): self.v = v\n    def __enter__(self): return self.v\n    def __exit__(self, *a): pass\nwith Ctx(42) as val:\n    result = val * 2\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=84" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class Ctx:\n    def __init__(self, v): self.v = v\n    def __enter__(self): return self.v\n    def __exit__(self, *a): pass\nwith Ctx(100) as val:\n    result = val * 3\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=300" in nb_runner.get_output(2)
