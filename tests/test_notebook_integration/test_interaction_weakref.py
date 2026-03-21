"""
Batch 292: Weakref interaction tests.
Tests that editing objects tracked via weakrefs properly invalidates
downstream computations.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestWeakrefInteraction:
    """Test weakref patterns with cache invalidation."""

    def test_weakref_basic_edit(self, nb_runner):
        """Editing the referent object should invalidate downstream."""
        nb_runner.create_notebook([
            "import weakref",
            "class Data:\n    def __init__(self, val):\n        self.val = val",
            "obj = Data(42)\nref = weakref.ref(obj)",
            "val = ref().val",
            "print(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "val=42" in out

        nb_runner.set_cell_source(3, "obj = Data(99)\nref = weakref.ref(obj)")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "val=99" in out

    def test_weakvaluedict_edit(self, nb_runner):
        """Editing objects in a WeakValueDictionary should propagate."""
        nb_runner.create_notebook([
            "import weakref",
            "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price",
            "a = Item('widget', 10)\nb = Item('gadget', 20)\ncache_dict = weakref.WeakValueDictionary()\ncache_dict['a'] = a\ncache_dict['b'] = b",
            "total = sum(cache_dict[k].price for k in cache_dict)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=30" in out

        nb_runner.set_cell_source(3, "a = Item('widget', 100)\nb = Item('gadget', 200)\ncache_dict = weakref.WeakValueDictionary()\ncache_dict['a'] = a\ncache_dict['b'] = b")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=300" in out
