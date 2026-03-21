"""
Interaction test: copy module deepcopy with custom classes.
Tests copy.copy vs copy.deepcopy behavior with nested structures,
__copy__/__deepcopy__ protocols, and cross-cell independence.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCopyDeepcopyCross:
    """Test copy/deepcopy across cells with custom objects."""

    def test_copy_vs_deepcopy(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create nested structure
            "import copy\ninner = [1, 2, 3]\nouter = {'data': inner, 'label': 'v1'}\nshallow = copy.copy(outer)\ndeep = copy.deepcopy(outer)\nprint(f'shallow_is_same_inner={shallow[\"data\"] is inner}')\nprint(f'deep_is_same_inner={deep[\"data\"] is inner}')",
            # Cell 2: mutate inner, check propagation
            "inner.append(4)\nshallow_len = len(shallow['data'])\ndeep_len = len(deep['data'])\nprint(f'shallow_len={shallow_len}')\nprint(f'deep_len={deep_len}')",
            # Cell 3: verify labels independent
            "shallow['label'] = 'v2'\nprint(f'outer_label={outer[\"label\"]}')\nprint(f'shallow_label={shallow[\"label\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "shallow_is_same_inner=True" in out1
        assert "deep_is_same_inner=False" in out1
        out2 = nb_runner.get_output(2)
        assert "shallow_len=4" in out2
        assert "deep_len=3" in out2
        out3 = nb_runner.get_output(3)
        assert "outer_label=v1" in out3
        assert "shallow_label=v2" in out3

    def test_copy_custom_class_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\nclass Box:\n    def __init__(self, items):\n        self.items = list(items)\n    def __copy__(self):\n        return Box(self.items)\n    def __deepcopy__(self, memo):\n        return Box(copy.deepcopy(self.items, memo))\nb = Box([10, 20])\ndc = copy.deepcopy(b)\nprint(f'orig={b.items}')\nprint(f'deep={dc.items}')",
            "dc.items.append(30)\nprint(f'orig_after={b.items}')\nprint(f'deep_after={dc.items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "orig=[10, 20]" in out1
        out2 = nb_runner.get_output(2)
        assert "orig_after=[10, 20]" in out2
        assert "deep_after=[10, 20, 30]" in out2

        # Edit items
        nb_runner.set_cell_source(1, "import copy\nclass Box:\n    def __init__(self, items):\n        self.items = list(items)\n    def __copy__(self):\n        return Box(self.items)\n    def __deepcopy__(self, memo):\n        return Box(copy.deepcopy(self.items, memo))\nb = Box([100, 200])\ndc = copy.deepcopy(b)\nprint(f'orig={b.items}')\nprint(f'deep={dc.items}')")
        nb_runner.run_cells([1, 2])
        out2b = nb_runner.get_output(2)
        assert "orig_after=[100, 200]" in out2b
        assert "deep_after=[100, 200, 30]" in out2b

    def test_copy_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\nsrc = {'a': [1], 'b': [2]}\ncloned = copy.deepcopy(src)\nprint(f'cloned_keys={sorted(cloned.keys())}')",
            "total = sum(v[0] for v in cloned.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cloned_keys=['a', 'b']" in nb_runner.get_output(1)
        assert "total=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)
