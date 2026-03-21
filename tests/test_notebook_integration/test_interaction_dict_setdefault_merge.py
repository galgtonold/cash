"""
Interaction test: dict.setdefault and dict.update with merge operator.
Tests dict.setdefault for conditional insertion, dict update patterns,
PEP 584 merge operator (|), and cross-cell dict manipulation.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictSetdefaultMerge:
    """Test dict.setdefault and merge operator across cells."""

    def test_dict_setdefault(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: setdefault
            "d = {'a': 1, 'b': 2}\nd.setdefault('c', 3)\nd.setdefault('a', 99)  # doesn't change existing\nprint(f'd={d}')",
            # Cell 2: merge operator |
            "extra = {'d': 4, 'e': 5}\nmerged = d | extra\nprint(f'merged={merged}')\nprint(f'original_unchanged={len(d) == 3}')",
            # Cell 3: |= update
            "d2 = d.copy()\nd2 |= {'f': 6, 'a': 100}\nprint(f'd2={d2}')\nprint(f'a_updated={d2[\"a\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'a': 1" in out1
        assert "'c': 3" in out1
        out2 = nb_runner.get_output(2)
        assert "'d': 4" in out2
        assert "'e': 5" in out2
        assert "original_unchanged=True" in out2
        out3 = nb_runner.get_output(3)
        assert "a_updated=100" in out3

    def test_dict_setdefault_edit(self, nb_runner):
        nb_runner.create_notebook([
            "counts = {}\nfor word in ['hello', 'world', 'hello']:\n    counts.setdefault(word, 0)\n    counts[word] += 1\nprint(f'counts={counts}')",
            "most_common = max(counts, key=counts.get)\nprint(f'most_common={most_common}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "most_common=hello" in nb_runner.get_output(2)

        # Edit words
        nb_runner.set_cell_source(1, "counts = {}\nfor word in ['foo', 'bar', 'foo', 'bar', 'bar']:\n    counts.setdefault(word, 0)\n    counts[word] += 1\nprint(f'counts={counts}')")
        nb_runner.run_cells([1, 2])
        assert "most_common=bar" in nb_runner.get_output(2)

    def test_dict_merge_cache(self, nb_runner):
        nb_runner.create_notebook([
            "a = {'x': 1}\nb = {'y': 2}\nresult = a | b\nprint(f'result={result}')",
            "total = sum(result.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)
