"""
Interaction test: itertools chain with generators and lazy evaluation.
Tests chain with generator expressions, lazy flattening,
and cross-cell lazy pipeline patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainLazyGenerator:
    """Test itertools chain with generators across cells."""

    def test_chain_lazy(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: chain generators
            "from itertools import chain\ndef gen_range(start, end):\n    for i in range(start, end):\n        yield i * i\n\nchained = list(chain(gen_range(1, 4), gen_range(4, 7)))\nprint(f'chained={chained}')",
            # Cell 2: chain with filter
            "evens = [x for x in chained if x % 2 == 0]\nprint(f'evens={evens}')",
            # Cell 3: sum from chained
            "total = sum(chained)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        # 1, 4, 9, 16, 25, 36
        assert "chained=[1, 4, 9, 16, 25, 36]" in out1
        out2 = nb_runner.get_output(2)
        assert "evens=[4, 16, 36]" in out2
        out3 = nb_runner.get_output(3)
        assert "total=91" in out3

    def test_chain_lazy_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import chain\na = [1, 2, 3]\nb = [4, 5, 6]\nresult = list(chain(a, b))\nprint(f'result={result}')",
            "length = len(result)\nprint(f'length={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length=6" in nb_runner.get_output(2)

        # Edit to add more
        nb_runner.set_cell_source(1, "from itertools import chain\na = [1, 2, 3]\nb = [4, 5, 6]\nc = [7, 8]\nresult = list(chain(a, b, c))\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "length=8" in nb_runner.get_output(2)

    def test_chain_lazy_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import chain\nwords = list(chain(['a', 'b'], ['c', 'd'], ['e']))\nprint(f'words={words}')",
            "joined = ''.join(words)\nprint(f'joined={joined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['a', 'b', 'c', 'd', 'e']" in nb_runner.get_output(1)
        assert "joined=abcde" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "joined=abcde" in nb_runner.get_output(2)
