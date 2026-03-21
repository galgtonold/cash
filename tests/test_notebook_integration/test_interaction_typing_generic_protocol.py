"""
Interaction test: typing module with TypeVar, Generic, Protocol.
Tests type annotation patterns used in data science code,
cross-cell generic class usage, and cache invalidation.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTypingGenericProtocol:
    """Test typing constructs across cells."""

    def test_typing_generic(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: generic stack
            "from typing import Generic, TypeVar, List\nT = TypeVar('T')\nclass Stack(Generic[T]):\n    def __init__(self):\n        self._items: List[T] = []\n    def push(self, item: T) -> None:\n        self._items.append(item)\n    def pop(self) -> T:\n        return self._items.pop()\n    def size(self) -> int:\n        return len(self._items)\n\ns = Stack()\ns.push(10)\ns.push(20)\ns.push(30)\nprint(f'size={s.size()}')",
            # Cell 2: use stack
            "top = s.pop()\nprint(f'top={top}')\nprint(f'size_after={s.size()}')",
            # Cell 3: string stack
            "ss = Stack()\nfor w in ['hello', 'world']:\n    ss.push(w)\nprint(f'str_size={ss.size()}')\nprint(f'str_top={ss.pop()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "size=3" in out1
        out2 = nb_runner.get_output(2)
        assert "top=30" in out2
        assert "size_after=2" in out2
        out3 = nb_runner.get_output(3)
        assert "str_size=2" in out3
        assert "str_top=world" in out3

    def test_typing_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: float\n    y: float\np = Point(3.0, 4.0)\nprint(f'point={p}')",
            "dist = (p.x ** 2 + p.y ** 2) ** 0.5\nprint(f'dist={dist}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "point=Point(x=3.0, y=4.0)" in nb_runner.get_output(1)
        assert "dist=5.0" in nb_runner.get_output(2)

        # Edit point
        nb_runner.set_cell_source(1, "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: float\n    y: float\np = Point(5.0, 12.0)\nprint(f'point={p}')")
        nb_runner.run_cells([1, 2])
        assert "point=Point(x=5.0, y=12.0)" in nb_runner.get_output(1)
        assert "dist=13.0" in nb_runner.get_output(2)

    def test_typing_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import Dict, Tuple\ndef make_pair(k: str, v: int) -> Tuple[str, int]:\n    return (k, v)\npair = make_pair('age', 25)\nprint(f'pair={pair}')",
            "key_val = f'{pair[0]}={pair[1]}'\nprint(f'kv={key_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pair=('age', 25)" in nb_runner.get_output(1)
        assert "kv=age=25" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "kv=age=25" in nb_runner.get_output(2)
