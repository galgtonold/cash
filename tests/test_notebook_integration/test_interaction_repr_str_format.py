"""Batch 453: class __repr__ and __str__ methods."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestReprStr:
    def test_repr_str(self, nb_runner):
        nb_runner.create_notebook([
            "class Coin:\n    def __init__(self, name, value):\n        self.name = name\n        self.value = value\n    def __repr__(self): return f'Coin({self.name!r}, {self.value})'\n    def __str__(self): return f'{self.name}=${self.value}'",
            "c = Coin('quarter', 0.25)\nr = repr(c)\ns = str(c)\nprint(f'repr={r} str={s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "repr=Coin('quarter', 0.25)" in out
        assert "str=quarter=$0.25" in out

    def test_format_method(self, nb_runner):
        nb_runner.create_notebook([
            "class Duration:\n    def __init__(self, seconds):\n        self.seconds = seconds\n    def __format__(self, spec):\n        if spec == 'hms':\n            h, rem = divmod(self.seconds, 3600)\n            m, s = divmod(rem, 60)\n            return f'{h}h{m}m{s}s'\n        return str(self.seconds)",
            "d = Duration(3661)\nprint(f'hms={d:hms} raw={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hms=1h1m1s" in nb_runner.get_output(2)
        assert "raw=3661" in nb_runner.get_output(2)

    def test_repr_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Tag:\n    def __init__(self, name): self.name = name\n    def __repr__(self): return f'Tag({self.name!r})'",
            "t = Tag('python')\nprint(f'tag={t!r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "tag=Tag('python')" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "t = Tag('rust')\nprint(f'tag={t!r}')")
        nb_runner.run_all()
        assert "tag=Tag('rust')" in nb_runner.get_output(2)
