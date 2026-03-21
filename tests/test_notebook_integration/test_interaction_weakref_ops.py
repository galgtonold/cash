"""
Interaction test: weakref module weak references.
Tests weakref.ref, finalize, WeakValueDictionary,
and cross-cell reference management patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestWeakrefOps:
    """Test weakref module across cells."""

    def test_weakref_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic weakref
            "import weakref\n\nclass MyObj:\n    def __init__(self, name):\n        self.name = name\n    def __repr__(self):\n        return f'MyObj({self.name})'\n\nobj = MyObj('test')\nref = weakref.ref(obj)\nprint(f'alive={ref() is not None}')\nprint(f'name={ref().name}')",
            # Cell 2: WeakValueDictionary
            "cache = weakref.WeakValueDictionary()\na = MyObj('alpha')\nb = MyObj('beta')\ncache['a'] = a\ncache['b'] = b\nprint(f'cache_len={len(cache)}')\nprint(f'a_name={cache[\"a\"].name}')",
            # Cell 3: finalize
            "cleanup_log = []\ndef on_finalize(name):\n    cleanup_log.append(f'cleaned:{name}')\n\nc = MyObj('gamma')\nfin = weakref.finalize(c, on_finalize, 'gamma')\nprint(f'alive_before={fin.alive}')\ndel c\nprint(f'alive_after={fin.alive}')\nprint(f'log={cleanup_log}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "alive=True" in out1
        assert "name=test" in out1
        out2 = nb_runner.get_output(2)
        assert "cache_len=2" in out2
        assert "a_name=alpha" in out2
        out3 = nb_runner.get_output(3)
        assert "alive_before=True" in out3
        assert "alive_after=False" in out3
        assert "cleaned:gamma" in out3

    def test_weakref_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref\nclass Item:\n    def __init__(self, val):\n        self.val = val\n\nitem = Item(42)\nref = weakref.ref(item)\nprint(f'val={ref().val}')",
            "doubled = ref().val * 2\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled=84" in nb_runner.get_output(2)

        # Edit item value
        nb_runner.set_cell_source(1, "import weakref\nclass Item:\n    def __init__(self, val):\n        self.val = val\n\nitem = Item(100)\nref = weakref.ref(item)\nprint(f'val={ref().val}')")
        nb_runner.run_cells([1, 2])
        assert "doubled=200" in nb_runner.get_output(2)

    def test_weakref_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import weakref\nclass Data:\n    def __init__(self, x):\n        self.x = x\n\nd = Data(7)\nwr = weakref.ref(d)\nprint(f'x={wr().x}')",
            "is_alive = wr() is not None\nprint(f'alive={is_alive}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=7" in nb_runner.get_output(1)
        assert "alive=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "alive=True" in nb_runner.get_output(2)
