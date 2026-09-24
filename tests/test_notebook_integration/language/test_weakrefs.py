"""weakref across cells."""

import textwrap

import pytest


# Weakref & memory patterns — cash caching with weak references and GC.
@pytest.mark.stress
class TestWeakrefBasics:
    """Test weak references and garbage collection."""

    def test_weakvalue_dict(self, nb_runner):
        """WeakValueDictionary pattern."""
        nb_runner.create_notebook(
            [
                "import weakref",
                textwrap.dedent("""\
                class CacheEntry:
                    def __init__(self, data):
                        self.data = data

                cache = weakref.WeakValueDictionary()
                entries = []
                for i in range(3):
                    e = CacheEntry(f"data_{i}")
                    cache[f"key_{i}"] = e
                    entries.append(e)  # keep strong refs
                print(f"cache_size={len(cache)}")
            """),
                textwrap.dedent("""\
                values = [cache[k].data for k in sorted(cache.keys())]
                print(f"values={values}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cache_size=3" in nb_runner.get_output(2)
        assert "values=['data_0', 'data_1', 'data_2']" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestWeakrefUsage:
    """weakref and weak references."""

    def test_weakref_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import weakref\nclass MyObj:\n    def __init__(self, val): self.val = val\nobj = MyObj(42)",
                "ref = weakref.ref(obj)\nalive = ref() is not None\nval = ref().val\nprint(f'alive={alive} val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "alive=True" in nb_runner.get_output(2)
        assert "val=42" in nb_runner.get_output(2)

    def test_weakref_dict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import weakref\nclass Item:\n    def __init__(self, name): self.name = name\nd = weakref.WeakValueDictionary()",
                "item = Item('test')\nd['key'] = item\nfound = 'key' in d\nname = d['key'].name\nprint(f'found={found} name={name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "found=True" in nb_runner.get_output(2)
        assert "name=test" in nb_runner.get_output(2)

    def test_weakref_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import weakref\nclass Box:\n    def __init__(self, v): self.v = v\nb = Box(10)",
                "r = weakref.ref(b)\nprint(f'v={r().v}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import weakref\nclass Box:\n    def __init__(self, v): self.v = v\nb = Box(99)")
        nb_runner.run_all()
        assert "v=99" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestWeakrefObjects:
    """weakref and weak references to objects."""

    def test_weakref_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import weakref",
                "class Obj:\n    def __init__(self, name): self.name = name\no = Obj('test')\nref = weakref.ref(o)\nprint(f'alive={ref() is not None} name={ref().name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "alive=True" in out
        assert "name=test" in out

    def test_weakvaluedict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import weakref",
                "class Item:\n    def __init__(self, val): self.val = val\ncache = weakref.WeakValueDictionary()\na = Item(10)\nb = Item(20)\ncache['a'] = a\ncache['b'] = b\nprint(f'len={len(cache)} a={cache[\"a\"].val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len=2" in out
        assert "a=10" in out

    def test_weakref_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import weakref",
                "class Obj:\n    def __init__(self, v): self.v = v\nx = Obj(42)\nr = weakref.ref(x)\nprint(f'v={r().v}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v=42" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "class Obj:\n    def __init__(self, v): self.v = v\nx = Obj(99)\nr = weakref.ref(x)\nprint(f'v={r().v}')"
        )
        nb_runner.run_all()
        assert "v=99" in nb_runner.get_output(2)


# Interaction test: weakref module weak references.
# Tests weakref.ref, finalize, WeakValueDictionary,
# and cross-cell reference management patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestWeakrefOps:
    """Test weakref module across cells."""

    def test_weakref_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic weakref
                "import weakref\n\nclass MyObj:\n    def __init__(self, name):\n        self.name = name\n    def __repr__(self):\n        return f'MyObj({self.name})'\n\nobj = MyObj('test')\nref = weakref.ref(obj)\nprint(f'alive={ref() is not None}')\nprint(f'name={ref().name}')",
                # Cell 2: WeakValueDictionary
                "cache = weakref.WeakValueDictionary()\na = MyObj('alpha')\nb = MyObj('beta')\ncache['a'] = a\ncache['b'] = b\nprint(f'cache_len={len(cache)}')\nprint(f'a_name={cache[\"a\"].name}')",
                # Cell 3: finalize
                "cleanup_log = []\ndef on_finalize(name):\n    cleanup_log.append(f'cleaned:{name}')\n\nc = MyObj('gamma')\nfin = weakref.finalize(c, on_finalize, 'gamma')\nprint(f'alive_before={fin.alive}')\ndel c\nprint(f'alive_after={fin.alive}')\nprint(f'log={cleanup_log}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import weakref\nclass Item:\n    def __init__(self, val):\n        self.val = val\n\nitem = Item(42)\nref = weakref.ref(item)\nprint(f'val={ref().val}')",
                "doubled = ref().val * 2\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled=84" in nb_runner.get_output(2)

        # Edit item value
        nb_runner.set_cell_source(
            1,
            "import weakref\nclass Item:\n    def __init__(self, val):\n        self.val = val\n\nitem = Item(100)\nref = weakref.ref(item)\nprint(f'val={ref().val}')",
        )
        nb_runner.run_cells([1, 2])
        assert "doubled=200" in nb_runner.get_output(2)

    def test_weakref_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import weakref\nclass Data:\n    def __init__(self, x):\n        self.x = x\n\nd = Data(7)\nwr = weakref.ref(d)\nprint(f'x={wr().x}')",
                "is_alive = wr() is not None\nprint(f'alive={is_alive}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=7" in nb_runner.get_output(1)
        assert "alive=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "alive=True" in nb_runner.get_output(2)


# Weakref interaction tests.
# Tests that editing objects tracked via weakrefs properly invalidates
# downstream computations.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestWeakrefInteraction:
    """Test weakref patterns with cache invalidation."""

    def test_weakref_basic_edit(self, nb_runner):
        """Editing the referent object should invalidate downstream."""
        nb_runner.create_notebook(
            [
                "import weakref",
                "class Data:\n    def __init__(self, val):\n        self.val = val",
                "obj = Data(42)\nref = weakref.ref(obj)",
                "val = ref().val",
                "print(f'val={val}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import weakref",
                "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price",
                "a = Item('widget', 10)\nb = Item('gadget', 20)\ncache_dict = weakref.WeakValueDictionary()\ncache_dict['a'] = a\ncache_dict['b'] = b",
                "total = sum(cache_dict[k].price for k in cache_dict)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=30" in out

        nb_runner.set_cell_source(
            3,
            "a = Item('widget', 100)\nb = Item('gadget', 200)\ncache_dict = weakref.WeakValueDictionary()\ncache_dict['a'] = a\ncache_dict['b'] = b",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=300" in out
