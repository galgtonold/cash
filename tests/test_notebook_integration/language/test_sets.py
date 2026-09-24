"""set and frozenset operations across cells."""

import pytest


@pytest.mark.stress
class TestSetOperationEdits:
    """Editing set operations."""

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_set_union(self, nb_runner):
        """Edit one set in a union."""
        nb_runner.create_notebook(
            [
                "a = {1, 2, 3}  # set A",
                "b = {3, 4, 5}  # set B",
                "result = a | b\nprint(f'result = {sorted(result)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5]" in nb_runner.get_output(3)

        # Change set B
        nb_runner.set_cell_source(2, "b = {10, 20}  # set B v2")
        nb_runner.run_all()
        assert "result = [1, 2, 3, 10, 20]" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_set_intersection(self, nb_runner):
        """Edit set intersection."""
        nb_runner.create_notebook(
            [
                "s1 = {1, 2, 3, 4, 5}  # intersection source 1",
                "s2 = {3, 4, 5, 6, 7}  # intersection source 2",
                "common = s1 & s2\nprint(f'common = {sorted(common)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common = [3, 4, 5]" in nb_runner.get_output(3)

        # Change s2
        nb_runner.set_cell_source(2, "s2 = {1, 5, 9}  # intersection source 2 v2")
        nb_runner.run_all()
        assert "common = [1, 5]" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_set_difference(self, nb_runner):
        """Edit set difference."""
        nb_runner.create_notebook(
            [
                "all_items = {1, 2, 3, 4, 5}  # diff source all",
                "remove = {2, 4}  # diff source remove",
                "remaining = all_items - remove\nprint(f'remaining = {sorted(remaining)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "remaining = [1, 3, 5]" in nb_runner.get_output(3)

        # Change what to remove
        nb_runner.set_cell_source(2, "remove = {1, 3, 5}  # diff source remove v2")
        nb_runner.run_all()
        assert "remaining = [2, 4]" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_set_chain(self, nb_runner):
        """Edit a chain of set operations."""
        nb_runner.create_notebook(
            [
                "x = {1, 2, 3}  # set chain x",
                "y = {2, 3, 4}  # set chain y",
                "z = {3, 4, 5}  # set chain z",
                "result = (x | y) & z\nprint(f'result = {sorted(result)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x|y = {1,2,3,4}, (x|y)&z = {3,4}
        assert "result = [3, 4]" in nb_runner.get_output(4)

        # Change z
        nb_runner.set_cell_source(3, "z = {1, 2}  # set chain z v2")
        nb_runner.run_all()
        # x|y = {1,2,3,4}, (x|y)&z = {1,2}
        assert "result = [1, 2]" in nb_runner.get_output(4)

    # Set operations and edit propagation.
    #
    # Tests set unions, intersections, differences with edits.
    @pytest.mark.timeout(90)
    def test_set_union_edit(self, nb_runner):
        """Edit one set in union, result updates."""
        nb_runner.create_notebook(
            [
                "a = {1, 2, 3, 4}",
                "b = {3, 4, 5, 6}",
                "union = sorted(a | b)\nprint(f'union = {union}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "union = [1, 2, 3, 4, 5, 6]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "a = {10, 20, 30}")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "10" in out
        assert "20" in out
        assert "30" in out

    @pytest.mark.timeout(90)
    def test_set_intersection_edit(self, nb_runner):
        """Edit set, intersection changes."""
        nb_runner.create_notebook(
            [
                "primes = {2, 3, 5, 7, 11, 13}",
                "evens = {2, 4, 6, 8, 10, 12}",
                "common = sorted(primes & evens)\nprint(f'common = {common}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common = [2]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "evens = {2, 4, 6, 8, 10, 12, 3, 7}")
        nb_runner.run_all()
        assert "common = [2, 3, 7]" in nb_runner.get_output(3)

    @pytest.mark.timeout(90)
    def test_set_difference_edit(self, nb_runner):
        """Edit set, difference changes."""
        nb_runner.create_notebook(
            [
                "all_items = {'a', 'b', 'c', 'd', 'e'}",
                "used = {'a', 'c'}",
                "unused = sorted(all_items - used)\nprint(f'unused = {unused}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unused = ['b', 'd', 'e']" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "used = {'a', 'b', 'c', 'd'}")
        nb_runner.run_all()
        assert "unused = ['e']" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSetOperations:
    """set operations - union, intersection, symmetric_difference."""

    def test_union_intersection(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = {1, 2, 3, 4}\nb = {3, 4, 5, 6}",
                "union = sorted(a | b)\ninter = sorted(a & b)\ndiff = sorted(a - b)\nprint(f'union={union} inter={inter} diff={diff}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "union=[1, 2, 3, 4, 5, 6]" in out
        assert "inter=[3, 4]" in out
        assert "diff=[1, 2]" in out

    def test_symmetric_difference(self, nb_runner):
        nb_runner.create_notebook(
            [
                "x = {10, 20, 30}\ny = {20, 30, 40}",
                "sym = sorted(x ^ y)\nprint(f'sym_diff={sym}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sym_diff=[10, 40]" in nb_runner.get_output(2)

    def test_set_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "s1 = {1, 2, 3}\ns2 = {2, 3, 4}",
                "common = sorted(s1 & s2)\nall_items = sorted(s1 | s2)\nprint(f'common={common} all={all_items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common=[2, 3]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "s1 = {5, 6, 7}\ns2 = {7, 8, 9}")
        nb_runner.run_all()
        assert "common=[7]" in nb_runner.get_output(2)
        assert "all=[5, 6, 7, 8, 9]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSetOperationsAdvanced:
    """set operations (union, intersection, difference, symmetric_difference)."""

    def test_set_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = {1, 2, 3, 4, 5}\nb = {4, 5, 6, 7, 8}",
                "union = sorted(a | b)\ninter = sorted(a & b)\ndiff = sorted(a - b)\nsym = sorted(a ^ b)\nprint(f'union={union} inter={inter} diff={diff} sym={sym}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "union=[1, 2, 3, 4, 5, 6, 7, 8]" in out
        assert "inter=[4, 5]" in out
        assert "diff=[1, 2, 3]" in out
        assert "sym=[1, 2, 3, 6, 7, 8]" in out

    def test_set_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "s1 = {'apple', 'banana', 'cherry'}",
                "s2 = {'banana', 'date'}\ncommon = sorted(s1 & s2)\nprint(f'common={common}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common=['banana']" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "s1 = {'banana', 'date', 'elderberry'}")
        nb_runner.run_all()
        assert "common=['banana', 'date']" in nb_runner.get_output(2)

    def test_frozenset_in_set(self, nb_runner):
        nb_runner.create_notebook(
            [
                "groups = {frozenset({1, 2}), frozenset({3, 4}), frozenset({1, 2})}",
                "count = len(groups)\nprint(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSetOpsIntersectionDiff:
    """set operations intersection difference symmetric."""

    def test_set_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = {1, 2, 3, 4, 5}\nb = {4, 5, 6, 7, 8}",
                "inter = sorted(a & b)\nunion = sorted(a | b)\ndiff = sorted(a - b)\nsym = sorted(a ^ b)\nprint(f'inter={inter} union={union}')\nprint(f'diff={diff} sym={sym}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "inter=[4, 5]" in out
        assert "union=[1, 2, 3, 4, 5, 6, 7, 8]" in out
        assert "diff=[1, 2, 3]" in out
        assert "sym=[1, 2, 3, 6, 7, 8]" in out

    def test_set_issubset(self, nb_runner):
        nb_runner.create_notebook(
            [
                "small = {1, 2}\nbig = {1, 2, 3, 4}",
                "sub = small.issubset(big)\nsup = big.issuperset(small)\ndisj = small.isdisjoint({5, 6})\nprint(f'sub={sub} sup={sup} disj={disj}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sub=True" in out
        assert "sup=True" in out
        assert "disj=True" in out

    def test_set_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "s = {1, 2, 3}",
                "result = sorted(s & {2, 3, 4})\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[2, 3]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "s = {10, 20, 30}")
        nb_runner.run_all()
        assert "result=[]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestFrozensetOps:
    """Test frozenset operation caching."""

    def test_frozenset_intersection(self, nb_runner):
        """Frozenset intersection, verify caching."""
        nb_runner.create_notebook(
            [
                "a = frozenset([1, 2, 3, 4, 5])",
                "b = frozenset([3, 4, 5, 6, 7])",
                "common = a & b\nresult = sorted(common)",
                "print(f'common={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "common=[3, 4, 5]" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "common=[3, 4, 5]" in out2

    def test_frozenset_union_edit(self, nb_runner):
        """Edit one frozenset, verify union updates."""
        nb_runner.create_notebook(
            [
                "s1 = frozenset(['a', 'b', 'c'])",
                "s2 = frozenset(['c', 'd', 'e'])",
                "merged = s1 | s2\ncount = len(merged)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=5" in out

        nb_runner.set_cell_source(2, "s2 = frozenset(['c', 'd', 'e', 'f'])")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=6" in out2

    def test_frozenset_as_dict_key(self, nb_runner):
        """Use frozensets as dict keys (hashable)."""
        nb_runner.create_notebook(
            [
                "key1 = frozenset([1, 2])\nkey2 = frozenset([3, 4])",
                "lookup = {key1: 'first', key2: 'second'}",
                "query = frozenset([1, 2])\nval = lookup[query]\nprint(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "val=first" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "val=first" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFrozensetDictKey:
    """frozenset as dict key and set operations."""

    def test_frozenset_as_key(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = frozenset({1, 2, 3})\nb = frozenset({3, 4, 5})",
                "d = {a: 'first', b: 'second'}\nresult = d[frozenset({1, 2, 3})]\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=first" in nb_runner.get_output(2)

    def test_frozenset_set_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = frozenset({1, 2, 3})\nb = frozenset({2, 3, 4})",
                "u = sorted(a | b)\ni = sorted(a & b)\nd = sorted(a - b)\nprint(f'u={u} i={i} d={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "u=[1, 2, 3, 4]" in nb_runner.get_output(2)
        assert "i=[2, 3]" in nb_runner.get_output(2)
        assert "d=[1]" in nb_runner.get_output(2)

    def test_frozenset_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "fs = frozenset({10, 20, 30})",
                "contains = 20 in fs\ncount = len(fs)\nprint(f'contains={contains} count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "contains=True" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "fs = frozenset({40, 50})")
        nb_runner.run_all()
        assert "contains=False" in nb_runner.get_output(2)
        assert "count=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFrozensetDictKeys:
    """frozenset operations as dict keys and set algebra."""

    def test_frozenset_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "fs = frozenset([1, 2, 3, 2, 1])",
                "size = len(fs)\nhas2 = 2 in fs\nhas5 = 5 in fs\nprint(f'size={size} has2={has2} has5={has5}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=3" in nb_runner.get_output(2)
        assert "has2=True" in nb_runner.get_output(2)
        assert "has5=False" in nb_runner.get_output(2)

    def test_frozenset_as_key(self, nb_runner):
        nb_runner.create_notebook(
            [
                "groups = {}\ngroups[frozenset([1, 2])] = 'pair'\ngroups[frozenset([3, 4, 5])] = 'triple'",
                "r1 = groups[frozenset([2, 1])]\nr2 = groups[frozenset([5, 3, 4])]\nprint(f'r1={r1} r2={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=pair" in nb_runner.get_output(2)
        assert "r2=triple" in nb_runner.get_output(2)

    def test_frozenset_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = frozenset([1, 2, 3])\nb = frozenset([3, 4, 5])",
                "union = sorted(a | b)\ninter = sorted(a & b)\nprint(f'union={union} inter={inter}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "union=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "a = frozenset([10, 20])\nb = frozenset([20, 30])")
        nb_runner.run_all()
        assert "union=[10, 20, 30]" in nb_runner.get_output(2)
        assert "inter=[20]" in nb_runner.get_output(2)
