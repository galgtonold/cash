"""dict operations across cells: merges, setdefault, views, inversion and nested merges."""

import textwrap

import pytest


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestDictEdits:
    """Dict values with cell edits."""

    def test_dict_creation_edit(self, nb_runner):
        """Edit a dict creation cell."""
        nb_runner.create_notebook(
            [
                "config = {'a': 1, 'b': 2}",
                "total = sum(config.values())\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "config = {'a': 10, 'b': 20, 'c': 30}")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_dict_access_edit(self, nb_runner):
        """Edit how a dict is accessed."""
        nb_runner.create_notebook(
            [
                "data = {'x': 10, 'y': 20, 'z': 30}",
                "val = data['x']\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "val = data['z']\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 30" in nb_runner.get_output(2)

    def test_nested_dict_edit(self, nb_runner):
        """Edit a nested dict."""
        nb_runner.create_notebook(
            [
                "data = {'outer': {'inner': 42}}",
                "val = data['outer']['inner']\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = {'outer': {'inner': 99}}")
        nb_runner.run_all()
        assert "val = 99" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.integration
class TestDictOperations:
    """Advanced dictionary patterns."""

    def test_nested_dict_update(self, nb_runner):
        """Deep merge of nested dicts."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def deep_merge(base, override):
                    result = base.copy()
                    for k, v in override.items():
                        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                            result[k] = deep_merge(result[k], v)
                        else:
                            result[k] = v
                    return result

                base = {'db': {'host': 'localhost', 'port': 5432}, 'debug': False}
                override = {'db': {'port': 3306}, 'debug': True, 'cache': True}
                merged = deep_merge(base, override)
            """),
                "print(f'merged={merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'host': 'localhost'" in out  # preserved from base
        assert "'port': 3306" in out  # overridden
        assert "'debug': True" in out
        assert "'cache': True" in out


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDictComprehensionEdits:
    """Editing dict comprehension patterns."""

    def test_edit_dict_comprehension_expression(self, nb_runner):
        """Edit the value expression in a dict comprehension."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']",
                "d = {k: i for i, k in enumerate(keys)}\nprint(f'd = {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = {'a': 0, 'b': 1, 'c': 2}" in nb_runner.get_output(2)

        # Change value expression
        nb_runner.set_cell_source(2, "d = {k: (i+1)*10 for i, k in enumerate(keys)}\nprint(f'd = {d}')")
        nb_runner.run_all()
        assert "d = {'a': 10, 'b': 20, 'c': 30}" in nb_runner.get_output(2)

    def test_edit_dict_source_keys(self, nb_runner):
        """Edit the source keys list."""
        nb_runner.create_notebook(
            [
                "names = ['alice', 'bob']  # dict source keys",
                "scores = {n: len(n) for n in names}\nprint(f'scores = {scores}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "scores = {'alice': 5, 'bob': 3}" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "names = ['charlie', 'dan', 'eve']  # dict source keys updated")
        nb_runner.run_all()
        assert "'charlie': 7" in nb_runner.get_output(2)
        assert "'eve': 3" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDictMergeEdits:
    """Editing dict merge patterns."""

    def test_edit_merge_operand(self, nb_runner):
        """Edit one dict in a merge operation."""
        nb_runner.create_notebook(
            [
                "base = {'x': 1, 'y': 2}  # base dict",
                "override = {'y': 20, 'z': 30}  # override dict",
                "merged = {**base, **override}\nprint(f'merged = {merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'x': 1" in nb_runner.get_output(3)
        assert "'y': 20" in nb_runner.get_output(3)
        assert "'z': 30" in nb_runner.get_output(3)

        # Change override
        nb_runner.set_cell_source(2, "override = {'y': 200, 'w': 400}  # override dict changed")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'y': 200" in out
        assert "'w': 400" in out

    def test_edit_defaultdict_factory(self, nb_runner):
        """Edit defaultdict usage."""
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "dd = defaultdict(int)\ndd['a'] += 1\ndd['b'] += 5\nprint(f'dd = {dict(dd)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        assert "'b': 5" in nb_runner.get_output(2)

        # Change factory and values
        nb_runner.set_cell_source(
            2,
            "dd = defaultdict(list)\ndd['a'].append(10)\ndd['b'].extend([20, 30])\nprint(f'dd = {dict(dd)}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': [10]" in out
        assert "'b': [20, 30]" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictMergePatterns:
    """Dict merge and update patterns with edit propagation."""

    def test_dict_union_operator(self, nb_runner):
        """Edit one dict in union, merged result updates."""
        nb_runner.create_notebook(
            [
                "defaults = {'color': 'blue', 'size': 10}",
                "overrides = {'size': 20, 'shape': 'circle'}",
                "config = defaults | overrides\nprint(f'config = {config}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'color': 'blue'" in out
        assert "'size': 20" in out
        assert "'shape': 'circle'" in out

        nb_runner.set_cell_source(1, "defaults = {'color': 'red', 'size': 10, 'weight': 5}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "'color': 'red'" in out2
        assert "'weight': 5" in out2

    def test_dict_comprehension_merge(self, nb_runner):
        """Edit source dict, comprehension downstream updates."""
        nb_runner.create_notebook(
            [
                "prices = {'apple': 1.0, 'banana': 0.5, 'cherry': 2.0}",
                "discount = 0.8",
                "sale = {k: round(v * discount, 2) for k, v in prices.items()}\nprint(f'sale = {sale}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'apple': 0.8" in out
        assert "'banana': 0.4" in out

        nb_runner.set_cell_source(2, "discount = 0.5")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "'apple': 0.5" in out2
        assert "'cherry': 1.0" in out2

    def test_nested_dict_edit(self, nb_runner):
        """Edit nested dict structure, downstream uses nested access."""
        nb_runner.create_notebook(
            [
                "db = {'users': {'alice': 30, 'bob': 25}, 'version': 1}",
                "names = list(db['users'].keys())\nages = list(db['users'].values())\nprint(f'names={names} ages={ages}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['alice', 'bob']" in nb_runner.get_output(2)
        assert "ages=[30, 25]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "db = {'users': {'charlie': 40, 'diana': 35, 'eve': 28}, 'version': 2}",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "charlie" in out2
        assert "40" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictMergeUpdate:
    """dict merge operators and update patterns."""

    def test_merge_operator(self, nb_runner):
        nb_runner.create_notebook(
            [
                "d1 = {'a': 1, 'b': 2}\nd2 = {'b': 20, 'c': 30}",
                "merged = d1 | d2\nprint(f'merged={merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        assert "'b': 20" in nb_runner.get_output(2)
        assert "'c': 30" in nb_runner.get_output(2)

    def test_dict_comprehension_merge(self, nb_runner):
        nb_runner.create_notebook(
            [
                "keys = ['x', 'y', 'z']\nvals = [10, 20, 30]",
                "d = {k: v for k, v in zip(keys, vals)}\nreversed_d = {v: k for k, v in d.items()}\nprint(f'd={d} rev={reversed_d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'x': 10" in out
        assert "10: 'x'" in out

    def test_dict_merge_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "base = {'mode': 'fast', 'verbose': True}",
                "override = {'verbose': False}\nfinal = base | override\nprint(f'mode={final[\"mode\"]} verbose={final[\"verbose\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mode=fast" in nb_runner.get_output(2)
        assert "verbose=False" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "base = {'mode': 'slow', 'verbose': True}")
        nb_runner.run_all()
        assert "mode=slow" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestRecursiveDictMerge:
    """Test recursive dict merge across cells."""

    def test_recursive_merge(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define merge function
                "def deep_merge(base, override):\n    result = base.copy()\n    for k, v in override.items():\n        if k in result and isinstance(result[k], dict) and isinstance(v, dict):\n            result[k] = deep_merge(result[k], v)\n        else:\n            result[k] = v\n    return result\nprint('deep_merge defined')",
                # Cell 2: merge configs
                "base = {'db': {'host': 'localhost', 'port': 5432}, 'debug': False, 'log': 'info'}\noverride = {'db': {'port': 3306, 'name': 'mydb'}, 'debug': True}\nmerged = deep_merge(base, override)\nprint(f'host={merged[\"db\"][\"host\"]}')\nprint(f'port={merged[\"db\"][\"port\"]}')\nprint(f'name={merged[\"db\"][\"name\"]}')\nprint(f'debug={merged[\"debug\"]}')\nprint(f'log={merged[\"log\"]}')",
                # Cell 3: count keys
                "total_keys = len(merged) + len(merged['db'])\nprint(f'total_keys={total_keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "host=localhost" in out2
        assert "port=3306" in out2
        assert "name=mydb" in out2
        assert "debug=True" in out2
        assert "log=info" in out2
        out3 = nb_runner.get_output(3)
        assert "total_keys=6" in out3

    def test_merge_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def deep_merge(base, override):\n    result = base.copy()\n    for k, v in override.items():\n        if k in result and isinstance(result[k], dict) and isinstance(v, dict):\n            result[k] = deep_merge(result[k], v)\n        else:\n            result[k] = v\n    return result\nprint('deep_merge defined')",
                "base = {'a': 1, 'b': {'x': 10}}\nover = {'b': {'y': 20}}\nm = deep_merge(base, over)\nprint(f'result={m}')",
                "b_keys = sorted(m['b'].keys())\nprint(f'b_keys={b_keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b_keys=['x', 'y']" in nb_runner.get_output(3)

        # Edit override
        nb_runner.set_cell_source(
            2,
            "base = {'a': 1, 'b': {'x': 10}}\nover = {'b': {'y': 20, 'z': 30}, 'c': 3}\nm = deep_merge(base, over)\nprint(f'result={m}')",
        )
        nb_runner.run_cells([2, 3])
        assert "b_keys=['x', 'y', 'z']" in nb_runner.get_output(3)

    def test_merge_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def deep_merge(a, b):\n    r = a.copy()\n    for k, v in b.items():\n        if k in r and isinstance(r[k], dict) and isinstance(v, dict):\n            r[k] = deep_merge(r[k], v)\n        else:\n            r[k] = v\n    return r\nprint('defined')",
                "cfg = deep_merge({'x': 1}, {'y': 2})\nprint(f'keys={sorted(cfg.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictAdvancedOps:
    """dict.setdefault, dict.update, and chained dict ops."""

    def test_setdefault_grouping(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = [('fruit', 'apple'), ('veg', 'carrot'), ('fruit', 'banana'), ('veg', 'pea')]",
                "groups = {}\nfor cat, item in items:\n    groups.setdefault(cat, []).append(item)\nprint(f'groups={dict(sorted(groups.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fruit" in nb_runner.get_output(2)
        assert "'apple', 'banana'" in nb_runner.get_output(2)

    def test_dict_update_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "base = {'a': 1, 'b': 2}",
                "overlay = {'b': 20, 'c': 30}\nmerged = {**base, **overlay}\nprint(f'merged={dict(sorted(merged.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "merged={'a': 1, 'b': 20, 'c': 30}" in nb_runner.get_output(2)
        # Edit base
        nb_runner.set_cell_source(1, "base = {'a': 100, 'b': 200}")
        nb_runner.run_all()
        assert "merged={'a': 100, 'b': 20, 'c': 30}" in nb_runner.get_output(2)

    def test_dict_pop_get(self, nb_runner):
        nb_runner.create_notebook(
            [
                "d = {'x': 10, 'y': 20, 'z': 30}",
                "val = d.get('w', -1)\nprint(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=-1" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictSetdefaultGet:
    """dict setdefault and get with defaults."""

    def test_setdefault(self, nb_runner):
        nb_runner.create_notebook(
            [
                "d = {'a': 1}",
                "d.setdefault('a', 99)\nd.setdefault('b', 42)\nprint(f'a={d[\"a\"]} b={d[\"b\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1" in nb_runner.get_output(2)
        assert "b=42" in nb_runner.get_output(2)

    def test_get_default(self, nb_runner):
        nb_runner.create_notebook(
            [
                "config = {'debug': True, 'port': 8080}",
                "debug = config.get('debug', False)\nhost = config.get('host', 'localhost')\nport = config.get('port', 3000)\nprint(f'debug={debug} host={host} port={port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "debug=True" in nb_runner.get_output(2)
        assert "host=localhost" in nb_runner.get_output(2)
        assert "port=8080" in nb_runner.get_output(2)

    def test_setdefault_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "groups = {}",
                "for item in ['a', 'b', 'a', 'c', 'b', 'a']:\n    groups.setdefault(item, 0)\n    groups[item] += 1\nprint(f'groups={groups}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 3" in nb_runner.get_output(2)
        assert "'b': 2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "groups = {}")
        nb_runner.set_cell_source(
            2,
            "for item in ['x', 'x', 'y']:\n    groups.setdefault(item, 0)\n    groups[item] += 1\nprint(f'groups={groups}')",
        )
        nb_runner.run_all()
        assert "'x': 2" in nb_runner.get_output(2)
        assert "'y': 1" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictSetdefaultMerge:
    """Test dict.setdefault and merge operator across cells."""

    def test_dict_setdefault(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: setdefault
                "d = {'a': 1, 'b': 2}\nd.setdefault('c', 3)\nd.setdefault('a', 99)  # doesn't change existing\nprint(f'd={d}')",
                # Cell 2: merge operator |
                "extra = {'d': 4, 'e': 5}\nmerged = d | extra\nprint(f'merged={merged}')\nprint(f'original_unchanged={len(d) == 3}')",
                # Cell 3: |= update
                "d2 = d.copy()\nd2 |= {'f': 6, 'a': 100}\nprint(f'd2={d2}')\nprint(f'a_updated={d2[\"a\"]}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "counts = {}\nfor word in ['hello', 'world', 'hello']:\n    counts.setdefault(word, 0)\n    counts[word] += 1\nprint(f'counts={counts}')",
                "most_common = max(counts, key=counts.get)\nprint(f'most_common={most_common}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "most_common=hello" in nb_runner.get_output(2)

        # Edit words
        nb_runner.set_cell_source(
            1,
            "counts = {}\nfor word in ['foo', 'bar', 'foo', 'bar', 'bar']:\n    counts.setdefault(word, 0)\n    counts[word] += 1\nprint(f'counts={counts}')",
        )
        nb_runner.run_cells([1, 2])
        assert "most_common=bar" in nb_runner.get_output(2)

    def test_dict_merge_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = {'x': 1}\nb = {'y': 2}\nresult = a | b\nprint(f'result={result}')",
                "total = sum(result.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictInversion:
    """dict inversion and bidirectional mapping."""

    def test_invert_dict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "original = {'a': 1, 'b': 2, 'c': 3}",
                "inverted = {v: k for k, v in original.items()}\nprint(f'inverted={dict(sorted(inverted.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "inverted={1: 'a', 2: 'b', 3: 'c'}" in nb_runner.get_output(2)

    def test_invert_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "mapping = {'red': '#FF0000', 'green': '#00FF00'}",
                "reverse = {v: k for k, v in mapping.items()}\ncolor = reverse.get('#FF0000', 'unknown')\nprint(f'color={color}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color=red" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "mapping = {'blue': '#0000FF', 'red': '#FF0000'}")
        nb_runner.run_all()
        assert "color=red" in nb_runner.get_output(2)

    def test_groupby_inversion(self, nb_runner):
        nb_runner.create_notebook(
            [
                "scores = {'Alice': 'A', 'Bob': 'B', 'Charlie': 'A', 'Diana': 'B'}",
                "grouped = {}\nfor name, grade in scores.items():\n    grouped.setdefault(grade, []).append(name)\nprint(f'grouped={dict(sorted(grouped.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'A': ['Alice', 'Charlie']" in out
        assert "'B': ['Bob', 'Diana']" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictViewIteration:
    """dictionary view objects and iteration edits."""

    def test_dict_views(self, nb_runner):
        nb_runner.create_notebook(
            [
                "d = {'x': 10, 'y': 20, 'z': 30}",
                "keys = sorted(d.keys())\nvals = sorted(d.values())\nitems = sorted(d.items())\nprint(f'keys={keys} vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "keys=['x', 'y', 'z']" in out
        assert "vals=[10, 20, 30]" in out

    def test_dict_iteration_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "prices = {'apple': 1.5, 'banana': 0.5, 'cherry': 3.0}",
                "expensive = {k: v for k, v in prices.items() if v > 1.0}\nprint(f'expensive={dict(sorted(expensive.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "expensive={'apple': 1.5, 'cherry': 3.0}" in nb_runner.get_output(2)
        # Edit prices
        nb_runner.set_cell_source(1, "prices = {'apple': 0.5, 'banana': 2.0, 'cherry': 0.3}")
        nb_runner.run_all()
        assert "expensive={'banana': 2.0}" in nb_runner.get_output(2)

    def test_dict_comprehension_filter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "scores = {'Alice': 85, 'Bob': 42, 'Charlie': 91, 'Diana': 38}",
                "passed = {k: v for k, v in scores.items() if v >= 50}\ncount = len(passed)\nprint(f'passed={dict(sorted(passed.items()))} count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "passed={'Alice': 85, 'Charlie': 91}" in nb_runner.get_output(2)
        assert "count=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictViewsAsSets:
    """dict views (keys, values, items) as sets."""

    def test_keys_intersection(self, nb_runner):
        nb_runner.create_notebook(
            [
                "d1 = {'a': 1, 'b': 2, 'c': 3}\nd2 = {'b': 20, 'c': 30, 'd': 40}",
                "common = sorted(d1.keys() & d2.keys())\nonly_d1 = sorted(d1.keys() - d2.keys())\nprint(f'common={common} only_d1={only_d1}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common=['b', 'c']" in nb_runner.get_output(2)
        assert "only_d1=['a']" in nb_runner.get_output(2)

    def test_items_as_set(self, nb_runner):
        nb_runner.create_notebook(
            [
                "d1 = {'a': 1, 'b': 2}\nd2 = {'a': 1, 'b': 3}",
                "common_items = sorted(d1.items() & d2.items())\nprint(f'common_items={common_items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common_items=[('a', 1)]" in nb_runner.get_output(2)

    def test_dict_views_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "d = {'x': 10, 'y': 20, 'z': 30}",
                "vals = sorted(d.values())\nkeys = sorted(d.keys())\nprint(f'keys={keys} vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y', 'z']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "d = {'a': 100, 'b': 200}")
        nb_runner.run_all()
        assert "keys=['a', 'b']" in nb_runner.get_output(2)
        assert "vals=[100, 200]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestTupleDictOutputInteraction:
    """Test tuple/dict output patterns with cache invalidation."""

    def test_tuple_stats_edit(self, nb_runner):
        """Editing data for tuple-returning stats function."""
        nb_runner.create_notebook(
            [
                "def calc_stats(d):\n    return min(d), max(d), sum(d) / len(d)",
                "vals = [10, 20, 30, 40, 50]",
                "lo, hi, avg = calc_stats(vals)",
                "print(f'lo={lo},hi={hi},avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "lo=10,hi=50,avg=30.0" in out

        nb_runner.set_cell_source(2, "vals = [100, 200, 300]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "lo=100,hi=300,avg=200.0" in out

    def test_dict_analyze_edit(self, nb_runner):
        """Editing data for dict-returning analysis function."""
        nb_runner.create_notebook(
            [
                "def word_info(text):\n    w = text.split()\n    return {'wc': len(w), 'uw': len(set(w))}",
                "text = 'hello world hello python'",
                "info = word_info(text)",
                "print(f\"wc={info['wc']},uw={info['uw']}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "wc=4,uw=3" in out

        nb_runner.set_cell_source(2, "text = 'one two three four five'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "wc=5,uw=5" in out
