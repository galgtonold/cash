"""
Batch 40: Weakref, copy, and memory management patterns across cells.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestCopyPatterns:
    """Test shallow/deep copy across cells."""

    def test_shallow_copy_independence(self, nb_runner):
        """Shallow copy creates independent top-level container."""
        nb_runner.create_notebook([
            "import copy",
            "original = [1, 2, [3, 4]]",
            textwrap.dedent("""\
                shallow = copy.copy(original)
                shallow[0] = 99
                print(f"orig={original[0]} copy={shallow[0]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "orig=1 copy=99" in nb_runner.get_output(3)

    def test_deep_copy_full_independence(self, nb_runner):
        """Deep copy creates fully independent copy."""
        nb_runner.create_notebook([
            "import copy",
            "original = {'a': [1, 2, 3], 'b': {'nested': True}}",
            textwrap.dedent("""\
                deep = copy.deepcopy(original)
                deep['a'].append(4)
                deep['b']['nested'] = False
                print(f"orig_a={original['a']} deep_a={deep['a']}")
                print(f"orig_nested={original['b']['nested']} deep_nested={deep['b']['nested']}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "orig_a=[1, 2, 3]" in output
        assert "deep_a=[1, 2, 3, 4]" in output

    def test_copy_propagation_on_change(self, nb_runner):
        """Change original → copy is independent."""
        nb_runner.create_notebook([
            "import copy",
            "data = [10, 20, 30]",
            "snapshot = copy.deepcopy(data)",
            textwrap.dedent("""\
                print(f"data={data} snap={snapshot}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "data=[10, 20, 30] snap=[10, 20, 30]" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "data = [100, 200, 300]")
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "data=[100, 200, 300]" in output
        assert "snap=[100, 200, 300]" in output


class TestDefaultDictPatterns:
    """Test collections.defaultdict across cells."""

    def test_defaultdict_int(self, nb_runner):
        """defaultdict(int) for counting."""
        nb_runner.create_notebook([
            "from collections import defaultdict",
            textwrap.dedent("""\
                counts = defaultdict(int)
                words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
                for w in words:
                    counts[w] += 1
            """),
            textwrap.dedent("""\
                print(dict(sorted(counts.items())))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'apple': 3" in output
        assert "'banana': 2" in output

    def test_defaultdict_list(self, nb_runner):
        """defaultdict(list) for grouping."""
        nb_runner.create_notebook([
            "from collections import defaultdict",
            textwrap.dedent("""\
                groups = defaultdict(list)
                items = [('a', 1), ('b', 2), ('a', 3), ('b', 4)]
                for k, v in items:
                    groups[k].append(v)
            """),
            "print(dict(sorted(groups.items())))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'a': [1, 3]" in output
        assert "'b': [2, 4]" in output


class TestOrderedDictPatterns:
    """Test collections.OrderedDict across cells."""

    def test_ordered_dict(self, nb_runner):
        """OrderedDict preserves insertion order."""
        nb_runner.create_notebook([
            "from collections import OrderedDict",
            textwrap.dedent("""\
                od = OrderedDict()
                od['c'] = 3
                od['a'] = 1
                od['b'] = 2
            """),
            textwrap.dedent("""\
                keys = list(od.keys())
                print(keys)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['c', 'a', 'b']" in nb_runner.get_output(3)


class TestDequePatterns:
    """Test collections.deque across cells."""

    def test_deque_operations(self, nb_runner):
        """Deque operations across cells."""
        nb_runner.create_notebook([
            "from collections import deque",
            textwrap.dedent("""\
                dq = deque([1, 2, 3], maxlen=5)
                dq.append(4)
                dq.appendleft(0)
            """),
            textwrap.dedent("""\
                print(f"deque={list(dq)} len={len(dq)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "deque=[0, 1, 2, 3, 4]" in nb_runner.get_output(3)

    def test_deque_as_circular_buffer(self, nb_runner):
        """Deque used as circular buffer."""
        nb_runner.create_notebook([
            "from collections import deque",
            textwrap.dedent("""\
                buffer = deque(maxlen=3)
                for i in range(6):
                    buffer.append(i)
            """),
            textwrap.dedent("""\
                print(f"buffer={list(buffer)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "buffer=[3, 4, 5]" in nb_runner.get_output(3)
