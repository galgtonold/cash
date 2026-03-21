"""Batch 230 – Iterator and protocol interaction tests.

Tests editing cells with custom iterators, context managers,
and protocol-based patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestIteratorProtocolEdits:
    """Editing custom iterators and protocol patterns."""

    def test_edit_custom_range_iterator(self, nb_runner):
        """Edit a custom range-like iterator."""
        nb_runner.create_notebook([
            "class MyRange:\n    def __init__(self, start, stop):\n        self.start = start\n        self.stop = stop\n    def __iter__(self):\n        current = self.start\n        while current < self.stop:\n            yield current\n            current += 1",
            "items = list(MyRange(0, 5))\nprint(f'items = {items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change step to 2
        nb_runner.set_cell_source(1, "class MyRange:\n    def __init__(self, start, stop):\n        self.start = start\n        self.stop = stop\n    def __iter__(self):\n        current = self.start\n        while current < self.stop:\n            yield current\n            current += 2")
        nb_runner.run_all()
        assert "items = [0, 2, 4]" in nb_runner.get_output(2)

    @pytest.mark.xfail(reason="Custom mutation methods (s.push()) not in MUTATING_METHODS; cache hits stale after cell edit. Needs mutation detector enhancement.")
    def test_edit_container_with_len(self, nb_runner):
        """Edit a custom container with __len__ and __getitem__."""
        nb_runner.create_notebook([
            "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def __len__(self):\n        return len(self._items)\n    def __repr__(self):\n        return f'Stack({self._items})'",
            "s = Stack()\ns.push(1)\ns.push(2)\ns.push(3)\nprint(f'len={len(s)} repr={s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3" in nb_runner.get_output(2)
        assert "Stack([1, 2, 3])" in nb_runner.get_output(2)

        # Edit to add only 2 items
        nb_runner.set_cell_source(2, "s = Stack()\ns.push(10)\ns.push(20)\nprint(f'len={len(s)} repr={s}')")
        nb_runner.run_all()
        assert "len=2" in nb_runner.get_output(2)
        assert "Stack([10, 20])" in nb_runner.get_output(2)

    def test_edit_callable_class(self, nb_runner):
        """Edit a callable class (__call__)."""
        nb_runner.create_notebook([
            "class Transformer:\n    def __init__(self, scale):\n        self.scale = scale\n    def __call__(self, x):\n        return x * self.scale",
            "t = Transformer(3)\nresult = t(7)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(2)

        # Edit scale
        nb_runner.set_cell_source(2, "t = Transformer(5)\nresult = t(7)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 35" in nb_runner.get_output(2)
