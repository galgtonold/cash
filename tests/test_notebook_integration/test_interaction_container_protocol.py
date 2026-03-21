"""Batch 396: class __contains__, __len__, __getitem__ protocol."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestContainerProtocol:
    def test_container_protocol(self, nb_runner):
        nb_runner.create_notebook([
            "class Bag:\n    def __init__(self, items):\n        self._items = list(items)\n    def __contains__(self, item):\n        return item in self._items\n    def __len__(self):\n        return len(self._items)\n    def __getitem__(self, idx):\n        return self._items[idx]",
            "b = Bag([10, 20, 30])\nhas_20 = 20 in b\nlength = len(b)\nfirst = b[0]\nprint(f'has_20={has_20} length={length} first={first}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "has_20=True length=3 first=10" in nb_runner.get_output(2)

    def test_container_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Stack:\n    def __init__(self):\n        self._data = []\n    def push(self, val):\n        self._data.append(val)\n    def __len__(self):\n        return len(self._data)\n    def __getitem__(self, idx):\n        return self._data[idx]",
            "s = Stack()\nfor v in [1, 2, 3]:\n    s.push(v)\nresult = f'{len(s)},{s[-1]}'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3,3" in nb_runner.get_output(2)
        # Edit to push different values
        nb_runner.set_cell_source(2, "s = Stack()\nfor v in [10, 20, 30, 40]:\n    s.push(v)\nresult = f'{len(s)},{s[-1]}'\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=4,40" in nb_runner.get_output(2)

    def test_bool_protocol(self, nb_runner):
        nb_runner.create_notebook([
            "class NonEmpty:\n    def __init__(self, items):\n        self.items = items\n    def __bool__(self):\n        return len(self.items) > 0",
            "a = NonEmpty([1])\nb = NonEmpty([])\nprint(f'a={bool(a)} b={bool(b)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=True b=False" in nb_runner.get_output(2)
