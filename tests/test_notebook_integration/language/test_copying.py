"""copy and deepcopy across cells."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


# Copy/deepcopy interaction tests.
# Tests that cache invalidation works correctly when objects are copied
# and the original is modified vs when the copy is modified.
@pytest.mark.integration
class TestCopyDeepcopyInteraction:
    """Test copy/deepcopy patterns with cache invalidation."""

    def test_shallow_copy_edit_original(self, nb_runner):
        """Editing original after shallow copy should invalidate original-dependent cells."""
        nb_runner.create_notebook(
            [
                "import copy\noriginal = [1, 2, 3]",
                "shallow = copy.copy(original)",
                "result = sum(original) + sum(shallow)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=12" in out

        # Edit original
        nb_runner.set_cell_source(1, "import copy\noriginal = [10, 20, 30]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=120" in out

    def test_deepcopy_edit_original(self, nb_runner):
        """Deepcopy should create independent object; editing original propagates."""
        nb_runner.create_notebook(
            [
                "import copy\ndata = {'a': [1, 2], 'b': [3, 4]}",
                "clone = copy.deepcopy(data)",
                "total = sum(data['a']) + sum(clone['b'])",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=10" in out

        # Edit data
        nb_runner.set_cell_source(1, "import copy\ndata = {'a': [10, 20], 'b': [30, 40]}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=100" in out

    def test_copy_chain_propagation(self, nb_runner):
        """Chain: original -> copy1 -> copy2; edit original propagates through."""
        nb_runner.create_notebook(
            [
                "import copy\nsrc = [1, 2, 3]",
                "c1 = copy.copy(src)",
                "c2 = copy.copy(c1)",
                "val = sum(c2)",
                "print(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "val=6" in out

        nb_runner.set_cell_source(1, "import copy\nsrc = [10, 20, 30]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "val=60" in out


# Interaction test: copy module deepcopy with custom classes.
# Tests copy.copy vs copy.deepcopy behavior with nested structures,
# __copy__/__deepcopy__ protocols, and cross-cell independence.
class TestCopyDeepcopyCross:
    """Test copy/deepcopy across cells with custom objects."""

    def test_copy_vs_deepcopy(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create nested structure
                "import copy\ninner = [1, 2, 3]\nouter = {'data': inner, 'label': 'v1'}\nshallow = copy.copy(outer)\ndeep = copy.deepcopy(outer)\nprint(f'shallow_is_same_inner={shallow[\"data\"] is inner}')\nprint(f'deep_is_same_inner={deep[\"data\"] is inner}')",
                # Cell 2: mutate inner, check propagation
                "inner.append(4)\nshallow_len = len(shallow['data'])\ndeep_len = len(deep['data'])\nprint(f'shallow_len={shallow_len}')\nprint(f'deep_len={deep_len}')",
                # Cell 3: verify labels independent
                "shallow['label'] = 'v2'\nprint(f'outer_label={outer[\"label\"]}')\nprint(f'shallow_label={shallow[\"label\"]}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import copy\nclass Box:\n    def __init__(self, items):\n        self.items = list(items)\n    def __copy__(self):\n        return Box(self.items)\n    def __deepcopy__(self, memo):\n        return Box(copy.deepcopy(self.items, memo))\nb = Box([10, 20])\ndc = copy.deepcopy(b)\nprint(f'orig={b.items}')\nprint(f'deep={dc.items}')",
                "dc.items.append(30)\nprint(f'orig_after={b.items}')\nprint(f'deep_after={dc.items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "orig=[10, 20]" in out1
        out2 = nb_runner.get_output(2)
        assert "orig_after=[10, 20]" in out2
        assert "deep_after=[10, 20, 30]" in out2

        # Edit items
        nb_runner.set_cell_source(
            1,
            "import copy\nclass Box:\n    def __init__(self, items):\n        self.items = list(items)\n    def __copy__(self):\n        return Box(self.items)\n    def __deepcopy__(self, memo):\n        return Box(copy.deepcopy(self.items, memo))\nb = Box([100, 200])\ndc = copy.deepcopy(b)\nprint(f'orig={b.items}')\nprint(f'deep={dc.items}')",
        )
        nb_runner.run_cells([1, 2])
        out2b = nb_runner.get_output(2)
        assert "orig_after=[100, 200]" in out2b
        assert "deep_after=[100, 200, 30]" in out2b

    def test_copy_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy\nsrc = {'a': [1], 'b': [2]}\ncloned = copy.deepcopy(src)\nprint(f'cloned_keys={sorted(cloned.keys())}')",
                "total = sum(v[0] for v in cloned.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cloned_keys=['a', 'b']" in nb_runner.get_output(1)
        assert "total=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)


class TestCopyDeepcopyNested:
    """copy deepcopy nested mutable objects."""

    def test_shallow_vs_deep(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy",
                "original = [[1, 2], [3, 4]]\nshallow = copy.copy(original)\ndeep = copy.deepcopy(original)\noriginal[0].append(99)\nprint(f'orig={original}')\nprint(f'shallow={shallow}')\nprint(f'deep={deep}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "orig=[[1, 2, 99], [3, 4]]" in out
        assert "shallow=[[1, 2, 99], [3, 4]]" in out
        assert "deep=[[1, 2], [3, 4]]" in out

    def test_deepcopy_dict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy",
                "d = {'a': [1, 2], 'b': {'c': 3}}\nd2 = copy.deepcopy(d)\nd['a'].append(99)\nd['b']['c'] = 999\nprint(f'd={d}')\nprint(f'd2={d2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d={'a': [1, 2, 99], 'b': {'c': 999}}" in out
        assert "d2={'a': [1, 2], 'b': {'c': 3}}" in out

    def test_copy_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy",
                "data = [1, 2, 3]\nclone = copy.deepcopy(data)\nprint(f'eq={data == clone} same={data is clone}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eq=True" in out
        assert "same=False" in out
        nb_runner.set_cell_source(
            2,
            "data = {'x': [1]}\nclone = copy.deepcopy(data)\ndata['x'].append(2)\nprint(f'data={data} clone={clone}')",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "data={'x': [1, 2]}" in out2
        assert "clone={'x': [1]}" in out2


class TestDeepVsShallowCopy:
    """copy.deepcopy vs shallow copy behaviors."""

    def test_shallow_vs_deep(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy\noriginal = [[1, 2], [3, 4]]",
                "shallow = copy.copy(original)\ndeep = copy.deepcopy(original)\noriginal[0][0] = 99\nshallow_changed = shallow[0][0]\ndeep_unchanged = deep[0][0]\nprint(f'shallow={shallow_changed} deep={deep_unchanged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "shallow=99" in nb_runner.get_output(2)
        assert "deep=1" in nb_runner.get_output(2)

    def test_deepcopy_dict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy\nd = {'a': [1, 2], 'b': {'c': 3}}",
                "d2 = copy.deepcopy(d)\nd['a'].append(99)\nd['b']['c'] = 999\nprint(f'orig_a={d[\"a\"]} copy_a={d2[\"a\"]} copy_c={d2[\"b\"][\"c\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "copy_a=[1, 2]" in out
        assert "copy_c=3" in out

    def test_copy_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy\ndata = [1, [2, 3]]",
                "cloned = copy.deepcopy(data)\ndata[1].append(4)\nprint(f'orig={data} clone={cloned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "clone=[1, [2, 3]]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import copy\ndata = [10, [20, 30]]")
        nb_runner.run_all()
        assert "clone=[10, [20, 30]]" in nb_runner.get_output(2)


class TestCopyDeepModify:
    """object copying (copy, deepcopy) with modifications."""

    def test_deep_copy(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy\noriginal = {'a': [1, 2], 'b': [3, 4]}",
                "deep = copy.deepcopy(original)\ndeep['a'].append(99)\nprint(f'original_a={original[\"a\"]}')\nprint(f'deep_a={deep[\"a\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "original_a=[1, 2]" in out
        assert "deep_a=[1, 2, 99]" in out

    def test_copy_edit_source(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import copy\ndata = [[1, 2], [3, 4]]",
                "cloned = copy.deepcopy(data)\ntotal = sum(sum(row) for row in cloned)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=10" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import copy\ndata = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "total=100" in nb_runner.get_output(2)
