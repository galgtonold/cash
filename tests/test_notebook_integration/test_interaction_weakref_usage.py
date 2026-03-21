"""Batch 457: weakref and weak references."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestWeakrefUsage:
    def test_weakref_basic(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref\nclass MyObj:\n    def __init__(self, val): self.val = val\nobj = MyObj(42)",
            "ref = weakref.ref(obj)\nalive = ref() is not None\nval = ref().val\nprint(f'alive={alive} val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "alive=True" in nb_runner.get_output(2)
        assert "val=42" in nb_runner.get_output(2)

    def test_weakref_dict(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref\nclass Item:\n    def __init__(self, name): self.name = name\nd = weakref.WeakValueDictionary()",
            "item = Item('test')\nd['key'] = item\nfound = 'key' in d\nname = d['key'].name\nprint(f'found={found} name={name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "found=True" in nb_runner.get_output(2)
        assert "name=test" in nb_runner.get_output(2)

    def test_weakref_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref\nclass Box:\n    def __init__(self, v): self.v = v\nb = Box(10)",
            "r = weakref.ref(b)\nprint(f'v={r().v}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import weakref\nclass Box:\n    def __init__(self, v): self.v = v\nb = Box(99)")
        nb_runner.run_all()
        assert "v=99" in nb_runner.get_output(2)
