"""Weakref & memory patterns — cash caching with weak references and GC."""

import textwrap

import pytest


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
