"""
Interaction test: dict subclass with custom default behavior.
Tests custom dict subclass with __missing__, __contains__ override,
and cross-cell dictionary operations.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictSubclassMissing:
    """Test dict subclass with __missing__ across cells."""

    def test_dict_missing(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define custom dict
            "class AutoDict(dict):\n    def __init__(self, factory, *args, **kw):\n        super().__init__(*args, **kw)\n        self._factory = factory\n    def __missing__(self, key):\n        self[key] = self._factory(key)\n        return self[key]\nprint('AutoDict defined')",
            # Cell 2: use with length factory
            "d = AutoDict(lambda k: len(k))\nprint(f'hello={d[\"hello\"]}')\nprint(f'hi={d[\"hi\"]}')\nprint(f'python={d[\"python\"]}')\nprint(f'keys={sorted(d.keys())}')",
            # Cell 3: aggregate
            "total_len = sum(d.values())\navg_len = total_len / len(d)\nprint(f'total={total_len}')\nprint(f'avg={avg_len:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "hello=5" in out2
        assert "hi=2" in out2
        assert "python=6" in out2
        out3 = nb_runner.get_output(3)
        assert "total=13" in out3
        assert "avg=4.3" in out3

    def test_dict_missing_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class AutoDict(dict):\n    def __init__(self, factory, *args, **kw):\n        super().__init__(*args, **kw)\n        self._factory = factory\n    def __missing__(self, key):\n        self[key] = self._factory(key)\n        return self[key]\nprint('AutoDict defined')",
            "d = AutoDict(lambda k: k.upper())\nd['cat']\nd['dog']\ncount = len(d)\nprint(f'vals={sorted(d.values())}')\nprint(f'count={count}')",
            "total_chars = sum(len(v) for v in d.values())\nprint(f'total_chars={total_chars}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=['CAT', 'DOG']" in nb_runner.get_output(2)
        assert "count=2" in nb_runner.get_output(2)
        assert "total_chars=6" in nb_runner.get_output(3)

        # Edit to access more keys
        nb_runner.set_cell_source(2, "d = AutoDict(lambda k: k.upper())\nd['cat']\nd['dog']\nd['bird']\ncount = len(d)\nprint(f'vals={sorted(d.values())}')\nprint(f'count={count}')")
        nb_runner.run_cells([2, 3])
        assert "vals=['BIRD', 'CAT', 'DOG']" in nb_runner.get_output(2)
        assert "count=3" in nb_runner.get_output(2)

    def test_dict_missing_cache(self, nb_runner):
        nb_runner.create_notebook([
            "class CountDict(dict):\n    def __init__(self):\n        super().__init__()\n        self.miss_count = 0\n    def __missing__(self, key):\n        self.miss_count += 1\n        self[key] = 0\n        return 0\nprint('CountDict defined')",
            "cd = CountDict()\ncd['a']\ncd['b']\ncd['a']  # not a miss\nprint(f'misses={cd.miss_count}')\nprint(f'keys={sorted(cd.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "misses=2" in nb_runner.get_output(2)
        assert "keys=['a', 'b']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "misses=2" in nb_runner.get_output(2)
