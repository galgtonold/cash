"""Batch 486: weakref and weak references to objects."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestWeakrefObjects:
    def test_weakref_basic(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref",
            "class Obj:\n    def __init__(self, name): self.name = name\no = Obj('test')\nref = weakref.ref(o)\nprint(f'alive={ref() is not None} name={ref().name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "alive=True" in out
        assert "name=test" in out

    def test_weakvaluedict(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref",
            "class Item:\n    def __init__(self, val): self.val = val\ncache = weakref.WeakValueDictionary()\na = Item(10)\nb = Item(20)\ncache['a'] = a\ncache['b'] = b\nprint(f'len={len(cache)} a={cache[\"a\"].val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len=2" in out
        assert "a=10" in out

    def test_weakref_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref",
            "class Obj:\n    def __init__(self, v): self.v = v\nx = Obj(42)\nr = weakref.ref(x)\nprint(f'v={r().v}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v=42" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class Obj:\n    def __init__(self, v): self.v = v\nx = Obj(99)\nr = weakref.ref(x)\nprint(f'v={r().v}')")
        nb_runner.run_all()
        assert "v=99" in nb_runner.get_output(2)
