"""ChainMap, deque, defaultdict, OrderedDict and UserDict across cells."""

import pytest

pytestmark = [pytest.mark.stress]


# Interaction test: ChainMap for layered dictionaries.
# Tests collections.ChainMap with multiple layers, new_child,
# parent traversal, and cross-cell config overlay patterns.
@pytest.mark.timeout(90)
class TestChainmapLayers:
    """Test ChainMap layered lookup across cells."""

    def test_chainmap_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create chain map
                "from collections import ChainMap\ndefaults = {'color': 'red', 'size': 'medium', 'style': 'bold'}\nuser_prefs = {'color': 'blue', 'font': 'arial'}\nconfig = ChainMap(user_prefs, defaults)\nprint(f'color={config[\"color\"]}')\nprint(f'size={config[\"size\"]}')\nprint(f'font={config[\"font\"]}')",
                # Cell 2: new_child for override
                "session = config.new_child({'color': 'green', 'zoom': 150})\nprint(f'session_color={session[\"color\"]}')\nprint(f'session_size={session[\"size\"]}')\nprint(f'original_color={config[\"color\"]}')",
                # Cell 3: list all unique keys
                "all_keys = sorted(set(session))\nprint(f'all_keys={all_keys}')\nprint(f'key_count={len(all_keys)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "color=blue" in out1
        assert "size=medium" in out1
        assert "font=arial" in out1
        out2 = nb_runner.get_output(2)
        assert "session_color=green" in out2
        assert "session_size=medium" in out2
        assert "original_color=blue" in out2
        out3 = nb_runner.get_output(3)
        assert "key_count=5" in out3

    def test_chainmap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap\nbase = {'a': 1, 'b': 2}\noverlay = {'b': 20, 'c': 30}\ncm = ChainMap(overlay, base)\nprint(f'a={cm[\"a\"]} b={cm[\"b\"]} c={cm[\"c\"]}')",
                "total = sum(cm.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=20 c=30" in nb_runner.get_output(1)
        # ChainMap.values() returns values from first occurrence of each key
        # Keys: a=1, b=20, c=30 => total=51
        assert "total=51" in nb_runner.get_output(2)

        # Edit overlay
        nb_runner.set_cell_source(
            1,
            "from collections import ChainMap\nbase = {'a': 1, 'b': 2}\noverlay = {'b': 200, 'c': 300}\ncm = ChainMap(overlay, base)\nprint(f'a={cm[\"a\"]} b={cm[\"b\"]} c={cm[\"c\"]}')",
        )
        nb_runner.run_cells([1, 2])
        assert "a=1 b=200 c=300" in nb_runner.get_output(1)
        assert "total=501" in nb_runner.get_output(2)

    def test_chainmap_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap\nenv = ChainMap({'PATH': '/usr/bin'}, {'HOME': '/home/user', 'PATH': '/bin'})\npath = env['PATH']\nprint(f'path={path}')",
                "has_home = 'HOME' in env\nprint(f'has_home={has_home}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "path=/usr/bin" in nb_runner.get_output(1)
        assert "has_home=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_home=True" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestChainMapPatterns:
    """collections.ChainMap usage patterns."""

    def test_chainmap_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap\ndefaults = {'color': 'red', 'size': 10}\noverrides = {'color': 'blue'}",
                "cm = ChainMap(overrides, defaults)\ncolor = cm['color']\nsize = cm['size']\nprint(f'color={color} size={size}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color=blue" in nb_runner.get_output(2)
        assert "size=10" in nb_runner.get_output(2)

    def test_chainmap_new_child(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap\nbase = {'a': 1, 'b': 2}\nlayer = {'b': 20}",
                'cm = ChainMap(layer, base)\nchild = cm.new_child({\'c\': 30})\nresult = dict(child)\nprint(f\'a={child["a"]} b={child["b"]} c={child["c"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a=1" in out
        assert "b=20" in out
        assert "c=30" in out

    def test_chainmap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap\nd1 = {'x': 10}\nd2 = {'y': 20}",
                "cm = ChainMap(d1, d2)\nkeys = sorted(cm.keys())\nprint(f'keys={keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import ChainMap\nd1 = {'a': 1}\nd2 = {'b': 2, 'c': 3}")
        nb_runner.run_all()
        assert "keys=['a', 'b', 'c']" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestChainMapProxy:
    """collections.ChainMap and MappingProxyType patterns."""

    def test_chainmap_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap\ndefaults = {'color': 'red', 'size': 'M'}\nuser = {'color': 'blue'}",
                "config = ChainMap(user, defaults)\nresult = dict(config)\nprint(f'result={dict(sorted(result.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result={'color': 'blue', 'size': 'M'}" in nb_runner.get_output(2)

    def test_chainmap_edit_user(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap\ndefaults = {'a': 1, 'b': 2}\noverrides = {'b': 20}",
                "cm = ChainMap(overrides, defaults)\nval_a = cm['a']\nval_b = cm['b']\nprint(f'a={val_a} b={val_b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=20" in nb_runner.get_output(2)
        # Edit overrides
        nb_runner.set_cell_source(
            1, "from collections import ChainMap\ndefaults = {'a': 1, 'b': 2}\noverrides = {'a': 100, 'b': 200}"
        )
        nb_runner.run_all()
        assert "a=100 b=200" in nb_runner.get_output(2)

    def test_mapping_proxy(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from types import MappingProxyType\ndata = {'x': 10, 'y': 20}\nproxy = MappingProxyType(data)",
                "vals = [proxy['x'], proxy['y']]\nprint(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=[10, 20]" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestChainMapOrderedDict:
    """collections ChainMap and OrderedDict move."""

    def test_chainmap_lookup(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap",
                "cmd_args = {'debug': True}\nenv_vars = {'debug': False, 'verbose': True}\ndefaults = {'debug': False, 'verbose': False, 'log': 'info'}\nconfig = ChainMap(cmd_args, env_vars, defaults)\nprint(f'debug={config[\"debug\"]} verbose={config[\"verbose\"]} log={config[\"log\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "debug=True" in out
        assert "verbose=True" in out
        assert "log=info" in out

    def test_ordereddict_move(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict",
                "od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])\nod.move_to_end('a')\norder1 = list(od.keys())\nod.move_to_end('c', last=False)\norder2 = list(od.keys())\nprint(f'order1={order1} order2={order2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "order1=['b', 'c', 'a']" in out
        assert "order2=['c', 'b', 'a']" in out

    def test_chainmap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import ChainMap",
                "a = {'x': 1}\nb = {'y': 2}\ncm = ChainMap(a, b)\nprint(f'x={cm[\"x\"]} y={cm[\"y\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=1" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "a = {'x': 99}\nb = {'y': 2}\ncm = ChainMap(a, b)\nprint(f'x={cm[\"x\"]} y={cm[\"y\"]}')"
        )
        nb_runner.run_all()
        assert "x=99" in nb_runner.get_output(2)


# defaultdict patterns with caching.
# Tests defaultdict(list), defaultdict(int), nested defaultdict, and edit propagation.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestDefaultdictPatterns:
    """Test defaultdict operation caching."""

    def test_defaultdict_list(self, nb_runner):
        """defaultdict(list) grouping pattern with caching."""
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "data = [('a', 1), ('b', 2), ('a', 3)]",
                "dd = defaultdict(list)\nfor k, v in data:\n    dd[k].append(v)\nresult = dict(sorted(dd.items()))",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result={'a': [1, 3], 'b': [2]}" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result={'a': [1, 3], 'b': [2]}" in out2

    def test_defaultdict_int_edit(self, nb_runner):
        """defaultdict(int) counting with edit."""
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "words = 'the cat sat on the mat the cat'.split()",
                "counts = defaultdict(int)\nfor w in words:\n    counts[w] += 1\nmost = max(counts, key=counts.get)",
                "print(f'most={most} count={counts[most]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "most=the" in out
        assert "count=3" in out

        nb_runner.set_cell_source(2, "words = 'a b a b a b a'.split()")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "most=a" in out2
        assert "count=4" in out2

    def test_defaultdict_nested(self, nb_runner):
        """Nested defaultdict pattern."""
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "data = [('US', 'NY', 100), ('US', 'CA', 200), ('UK', 'LN', 150)]",
                "nested = defaultdict(lambda: defaultdict(int))\nfor country, city, val in data:\n    nested[country][city] = val\nus_total = sum(nested['US'].values())",
                "print(f'us_total={us_total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "us_total=300" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "us_total=300" in out2


@pytest.mark.timeout(90)
class TestDefaultdictLambdaNesting:
    """defaultdict with lambda and complex nesting."""

    def test_defaultdict_lambda(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import defaultdict\ndd = defaultdict(lambda: 'unknown')",
                "dd['a'] = 'apple'\nr1 = dd['a']\nr2 = dd['b']\nprint(f'r1={r1} r2={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=apple" in nb_runner.get_output(2)
        assert "r2=unknown" in nb_runner.get_output(2)

    def test_nested_defaultdict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import defaultdict\ndef nested(): return defaultdict(int)\ndd = defaultdict(nested)",
                "dd['math']['hw1'] = 90\ndd['math']['hw2'] = 85\ndd['eng']['hw1'] = 92\nresult = dict(dd['math'])\nprint(f'math={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "math={'hw1': 90, 'hw2': 85}" in nb_runner.get_output(2)

    def test_defaultdict_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import defaultdict\ndata = [('a', 1), ('b', 2), ('a', 3)]",
                "dd = defaultdict(list)\nfor k, v in data:\n    dd[k].append(v)\nresult = dict(dd)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': [1, 3]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            1, "from collections import defaultdict\ndata = [('x', 10), ('y', 20), ('x', 30), ('y', 40)]"
        )
        nb_runner.run_all()
        assert "'x': [10, 30]" in nb_runner.get_output(2)
        assert "'y': [20, 40]" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestDefaultdictNestedFactory:
    """defaultdict nested and lambda factories."""

    def test_nested_defaultdict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "dd = defaultdict(lambda: defaultdict(int))\ndd['fruits']['apple'] += 3\ndd['fruits']['banana'] += 2\ndd['vegs']['carrot'] += 1\nprint(f'apple={dd[\"fruits\"][\"apple\"]} banana={dd[\"fruits\"][\"banana\"]} carrot={dd[\"vegs\"][\"carrot\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple=3" in out
        assert "banana=2" in out
        assert "carrot=1" in out

    def test_defaultdict_list(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "groups = defaultdict(list)\nfor name, dept in [('Alice','Eng'), ('Bob','Sales'), ('Carol','Eng'), ('Dave','Sales')]:\n    groups[dept].append(name)\nprint(f'eng={sorted(groups[\"Eng\"])} sales={sorted(groups[\"Sales\"])}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eng=['Alice', 'Carol']" in out
        assert "sales=['Bob', 'Dave']" in out

    def test_defaultdict_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "d = defaultdict(int)\nfor c in 'hello': d[c] += 1\nprint(f'l={d[\"l\"]} o={d[\"o\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "l=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "d = defaultdict(int)\nfor c in 'mississippi': d[c] += 1\nprint(f's={d[\"s\"]} p={d[\"p\"]}')"
        )
        nb_runner.run_all()
        assert "s=4" in nb_runner.get_output(2)
        assert "p=2" in nb_runner.get_output(2)


# collections.deque rotation and operation patterns with caching.
# Tests deque creation, rotate, appendleft, and edit propagation.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestDequeRotate:
    """Test collections.deque operation caching."""

    def test_deque_rotate_basic(self, nb_runner):
        """Create deque, rotate, verify caching."""
        nb_runner.create_notebook(
            [
                "from collections import deque",
                "d = deque([1, 2, 3, 4, 5])\nd.rotate(2)",
                "result = list(d)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result=[4, 5, 1, 2, 3]" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result=[4, 5, 1, 2, 3]" in out2

    def test_deque_maxlen_edit(self, nb_runner):
        """Deque with maxlen, edit propagation."""
        nb_runner.create_notebook(
            [
                "from collections import deque",
                "size = 3",
                "d = deque(range(10), maxlen=size)\nresult = list(d)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[7, 8, 9]" in out

        nb_runner.set_cell_source(2, "size = 5")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=[5, 6, 7, 8, 9]" in out2

    def test_deque_appendleft_pattern(self, nb_runner):
        """Build deque via appendleft in same cell."""
        nb_runner.create_notebook(
            [
                "from collections import deque",
                "items = [10, 20, 30]",
                "d = deque()\nfor item in items:\n    d.appendleft(item)\nresult = list(d)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[30, 20, 10]" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=[30, 20, 10]" in out2


@pytest.mark.timeout(90)
class TestDequeMaxlenRotate:
    """collections deque maxlen and rotate."""

    def test_deque_maxlen(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque",
                "d = deque(maxlen=3)\nfor i in range(5):\n    d.append(i)\nprint(f'd={list(d)} len={len(d)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=[2, 3, 4]" in out
        assert "len=3" in out

    def test_deque_rotate(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque",
                "d = deque([1, 2, 3, 4, 5])\nd.rotate(2)\nr1 = list(d)\nd.rotate(-3)\nr2 = list(d)\nprint(f'r1={r1} r2={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=[4, 5, 1, 2, 3]" in out
        assert "r2=[2, 3, 4, 5, 1]" in out

    def test_deque_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque",
                "d = deque([10, 20, 30])\nd.appendleft(5)\nprint(f'd={list(d)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=[5, 10, 20, 30]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = deque([10, 20, 30])\nd.extendleft([3, 2, 1])\nprint(f'd={list(d)}')")
        nb_runner.run_all()
        assert "d=[1, 2, 3, 10, 20, 30]" in nb_runner.get_output(2)


# Interaction test: deque appendleft and extendleft operations.
# Tests collections.deque with appendleft, extendleft, popleft,
# and cross-cell deque state management.
@pytest.mark.timeout(90)
class TestDequeAppendleftExtend:
    """Test deque appendleft and extendleft across cells."""

    def test_deque_left_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: build deque
                "from collections import deque\nd = deque([3, 4, 5])\nd.appendleft(2)\nd.appendleft(1)\nprint(f'deque={list(d)}')",
                # Cell 2: extendleft (note: order is reversed)
                "d2 = deque([4, 5, 6])\nd2.extendleft([3, 2, 1])  # 1 goes first, then 2, then 3\nprint(f'd2={list(d2)}')",
                # Cell 3: combine
                "merged = list(d) + list(d2)\nprint(f'merged_len={len(merged)}')\nprint(f'sorted={sorted(set(merged))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "deque=[1, 2, 3, 4, 5]" in out1
        out2 = nb_runner.get_output(2)
        assert "d2=[1, 2, 3, 4, 5, 6]" in out2
        out3 = nb_runner.get_output(3)
        assert "merged_len=11" in out3

    def test_deque_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nd = deque([1, 2, 3], maxlen=5)\nd.appendleft(0)\nprint(f'd={list(d)}')",
                "total = sum(d)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=[0, 1, 2, 3]" in nb_runner.get_output(1)
        assert "total=6" in nb_runner.get_output(2)

        # Edit cell 2 to also change
        nb_runner.set_cell_source(
            1,
            "from collections import deque\nd = deque([1, 2, 3], maxlen=5)\nd.appendleft(0)\nd.appendleft(-1)\nd.appendleft(-2)  # triggers drop from right\nprint(f'd={list(d)}')",
        )
        nb_runner.set_cell_source(2, "total = sum(d)  # recomputed\nprint(f'total={total}')")
        nb_runner.run_cells([1, 2])
        assert "d=[-2, -1, 0, 1, 2]" in nb_runner.get_output(1)
        assert "total=0" in nb_runner.get_output(2)

    def test_deque_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nd = deque(range(5))\nleft = d.popleft()\nright = d.pop()\nprint(f'left={left} right={right}')\nprint(f'remaining={list(d)}')",
                "mid = d[len(d)//2]\nprint(f'mid={mid}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "left=0 right=4" in nb_runner.get_output(1)
        assert "mid=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "mid=2" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestDequeRingBuffer:
    """collections.deque as ring buffer pattern."""

    def test_deque_maxlen(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nbuf = deque(maxlen=3)",
                "for i in range(5):\n    buf.append(i)\nresult = list(buf)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[2, 3, 4]" in nb_runner.get_output(2)

    def test_deque_rotate(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nd = deque([1, 2, 3, 4, 5])",
                "d.rotate(2)\nright = list(d)\nd.rotate(-4)\nleft = list(d)\nprint(f'right={right} left={left}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "right=[4, 5, 1, 2, 3]" in out

    def test_deque_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nd = deque([10, 20, 30])",
                "d.appendleft(5)\nd.append(35)\nresult = list(d)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[5, 10, 20, 30, 35]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import deque\nd = deque([100, 200])")
        nb_runner.run_all()
        assert "result=[5, 100, 200, 35]" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestDequeStackQueue:
    """collections.deque as stack/queue with maxlen."""

    def test_deque_as_queue(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nq = deque()\nfor i in range(5):\n    q.append(i)",
                "items = []\nwhile q:\n    items.append(q.popleft())\nprint(f'items={items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items=[0, 1, 2, 3, 4]" in nb_runner.get_output(2)

    def test_deque_maxlen_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nbuf = deque(maxlen=3)\nfor i in [10, 20, 30, 40, 50]:\n    buf.append(i)",
                "result = list(buf)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[30, 40, 50]" in nb_runner.get_output(2)
        # Edit maxlen
        nb_runner.set_cell_source(
            1, "from collections import deque\nbuf = deque(maxlen=2)\nfor i in [10, 20, 30, 40, 50]:\n    buf.append(i)"
        )
        nb_runner.run_all()
        assert "result=[40, 50]" in nb_runner.get_output(2)

    def test_deque_extend(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import deque\nd = deque([1, 2, 3])\nd.extendleft([10, 20])",
                "result = list(d)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[20, 10, 1, 2, 3]" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestOrderedDictBehavior:
    """collections.OrderedDict behavior."""

    def test_ordered_dict_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict\nod = OrderedDict([('b', 2), ('a', 1), ('c', 3)])",
                "keys = list(od.keys())\nvals = list(od.values())\nprint(f'keys={keys} vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['b', 'a', 'c']" in nb_runner.get_output(2)
        assert "vals=[2, 1, 3]" in nb_runner.get_output(2)

    def test_ordered_dict_move_to_end(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict\nod = OrderedDict([('x', 10), ('y', 20), ('z', 30)])",
                "od.move_to_end('x')\nresult = list(od.keys())\nprint(f'after_move={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "after_move=['y', 'z', 'x']" in nb_runner.get_output(2)

    def test_ordered_dict_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict\nod = OrderedDict([('a', 1), ('b', 2)])",
                "first = next(iter(od))\nlast = next(reversed(od))\nprint(f'first={first} last={last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=a last=b" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import OrderedDict\nod = OrderedDict([('z', 99), ('m', 50)])")
        nb_runner.run_all()
        assert "first=z last=m" in nb_runner.get_output(2)


# OrderedDict patterns with caching.
# Tests OrderedDict operations, move_to_end, and edit propagation.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestOrderedDictOps:
    """Test OrderedDict operation caching."""

    def test_ordered_dict_basic(self, nb_runner):
        """OrderedDict preserves insertion order with caching."""
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict",
                "od = OrderedDict([('b', 2), ('a', 1), ('c', 3)])",
                "keys = list(od.keys())\nprint(f'keys={keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "keys=['b', 'a', 'c']" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "keys=['b', 'a', 'c']" in out2

    def test_ordered_dict_edit(self, nb_runner):
        """Edit OrderedDict, verify order update."""
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict",
                "items = [('x', 10), ('y', 20), ('z', 30)]",
                "od = OrderedDict(items)\nfirst = list(od.keys())[0]",
                "print(f'first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "first=x" in out

        nb_runner.set_cell_source(2, "items = [('z', 30), ('x', 10), ('y', 20)]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "first=z" in out2


# Interaction test: OrderedDict equality and reversal.
# Tests OrderedDict order-sensitive equality, reversed() iteration,
# and cross-cell dict rebuilding.
@pytest.mark.timeout(90)
class TestOrderedDictEqualityReverse:
    """Test OrderedDict equality and reversal across cells."""

    def test_ordereddict_equality(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: order-sensitive equality
                "from collections import OrderedDict\nod1 = OrderedDict([('a', 1), ('b', 2), ('c', 3)])\nod2 = OrderedDict([('c', 3), ('b', 2), ('a', 1)])\nod3 = OrderedDict([('a', 1), ('b', 2), ('c', 3)])\nprint(f'eq_diff_order={od1 == od2}')\nprint(f'eq_same_order={od1 == od3}')",
                # Cell 2: reversed iteration
                "rev_keys = list(reversed(od1))\nrev_items = list(reversed(od1.items()))\nprint(f'rev_keys={rev_keys}')\nprint(f'rev_last_item={rev_items[0]}')",
                # Cell 3: rebuild from reversed
                "rebuilt = OrderedDict(reversed(list(od1.items())))\nprint(f'rebuilt_keys={list(rebuilt.keys())}')\nprint(f'rebuilt_eq_od2={rebuilt == od2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "eq_diff_order=False" in out1
        assert "eq_same_order=True" in out1
        out2 = nb_runner.get_output(2)
        assert "rev_keys=['c', 'b', 'a']" in out2
        assert "rev_last_item=('c', 3)" in out2
        out3 = nb_runner.get_output(3)
        assert "rebuilt_keys=['c', 'b', 'a']" in out3
        assert "rebuilt_eq_od2=True" in out3

    def test_ordereddict_eq_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict\ndata = OrderedDict(alpha=1, beta=2, gamma=3)\nprint(f'keys={list(data.keys())}')",
                "summary = '-'.join(data.keys())\nprint(f'summary={summary}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary=alpha-beta-gamma" in nb_runner.get_output(2)

        # Edit
        nb_runner.set_cell_source(
            1,
            "from collections import OrderedDict\ndata = OrderedDict(alpha=1, beta=2, gamma=3, delta=4)\nprint(f'keys={list(data.keys())}')",
        )
        nb_runner.run_cells([1, 2])
        assert "summary=alpha-beta-gamma-delta" in nb_runner.get_output(2)

    def test_ordereddict_eq_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict\nscores = OrderedDict(math=95, science=88, english=92)\nprint(f'count={len(scores)}')",
                "avg = sum(scores.values()) / len(scores)\nprint(f'avg={avg:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=91.7" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "avg=91.7" in nb_runner.get_output(2)


# Interaction test: collections.OrderedDict move_to_end and popitem.
# Tests OrderedDict ordering operations with move_to_end(last=False),
# popitem(last=True/False), and cross-cell state tracking.
@pytest.mark.timeout(90)
class TestOrderedDictMovePopitem:
    """Test OrderedDict move_to_end and popitem across cells."""

    def test_ordereddict_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create ordered dict
                "from collections import OrderedDict\nod = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])\nprint(f'keys={list(od.keys())}')",
                # Cell 2: move_to_end operations
                "od.move_to_end('a')  # move to end\nod.move_to_end('d', last=False)  # move to beginning\nordered = list(od.keys())\nprint(f'after_move={ordered}')",
                # Cell 3: popitem operations
                "last_item = od.popitem(last=True)\nfirst_item = od.popitem(last=False)\nremaining = list(od.keys())\nprint(f'popped_last={last_item}')\nprint(f'popped_first={first_item}')\nprint(f'remaining={remaining}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "keys=['a', 'b', 'c', 'd']" in out1
        out2 = nb_runner.get_output(2)
        assert "after_move=['d', 'b', 'c', 'a']" in out2
        out3 = nb_runner.get_output(3)
        assert "popped_last=('a', 1)" in out3
        assert "popped_first=('d', 4)" in out3
        assert "remaining=['b', 'c']" in out3

    def test_ordereddict_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict\nod = OrderedDict([('x', 10), ('y', 20), ('z', 30)])\nprint(f'keys={list(od.keys())}')",
                "od.move_to_end('x')\nresult = list(od.keys())\nprint(f'order={result}')",
                "vals = list(od.values())\nprint(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "order=['y', 'z', 'x']" in nb_runner.get_output(2)
        assert "vals=[20, 30, 10]" in nb_runner.get_output(3)

        # Edit initial dict
        nb_runner.set_cell_source(
            1,
            "from collections import OrderedDict\nod = OrderedDict([('x', 10), ('y', 20), ('z', 30), ('w', 40)])\nprint(f'keys={list(od.keys())}')",
        )
        nb_runner.run_cells([1, 2, 3])
        assert "order=['y', 'z', 'w', 'x']" in nb_runner.get_output(2)
        assert "vals=[20, 30, 40, 10]" in nb_runner.get_output(3)

    def test_ordereddict_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import OrderedDict\nod = OrderedDict(alpha=1, beta=2, gamma=3)\nprint(f'count={len(od)}')",
                "reversed_keys = list(reversed(od))\nprint(f'reversed={reversed_keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "reversed=['gamma', 'beta', 'alpha']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "reversed=['gamma', 'beta', 'alpha']" in nb_runner.get_output(2)


# Interaction test: dict subclass with custom default behavior.
# Tests custom dict subclass with __missing__, __contains__ override,
# and cross-cell dictionary operations.
@pytest.mark.timeout(90)
class TestDictSubclassMissing:
    """Test dict subclass with __missing__ across cells."""

    def test_dict_missing(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define custom dict
                "class AutoDict(dict):\n    def __init__(self, factory, *args, **kw):\n        super().__init__(*args, **kw)\n        self._factory = factory\n    def __missing__(self, key):\n        self[key] = self._factory(key)\n        return self[key]\nprint('AutoDict defined')",
                # Cell 2: use with length factory
                "d = AutoDict(lambda k: len(k))\nprint(f'hello={d[\"hello\"]}')\nprint(f'hi={d[\"hi\"]}')\nprint(f'python={d[\"python\"]}')\nprint(f'keys={sorted(d.keys())}')",
                # Cell 3: aggregate
                "total_len = sum(d.values())\navg_len = total_len / len(d)\nprint(f'total={total_len}')\nprint(f'avg={avg_len:.1f}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "class AutoDict(dict):\n    def __init__(self, factory, *args, **kw):\n        super().__init__(*args, **kw)\n        self._factory = factory\n    def __missing__(self, key):\n        self[key] = self._factory(key)\n        return self[key]\nprint('AutoDict defined')",
                "d = AutoDict(lambda k: k.upper())\nd['cat']\nd['dog']\ncount = len(d)\nprint(f'vals={sorted(d.values())}')\nprint(f'count={count}')",
                "total_chars = sum(len(v) for v in d.values())\nprint(f'total_chars={total_chars}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=['CAT', 'DOG']" in nb_runner.get_output(2)
        assert "count=2" in nb_runner.get_output(2)
        assert "total_chars=6" in nb_runner.get_output(3)

        # Edit to access more keys
        nb_runner.set_cell_source(
            2,
            "d = AutoDict(lambda k: k.upper())\nd['cat']\nd['dog']\nd['bird']\ncount = len(d)\nprint(f'vals={sorted(d.values())}')\nprint(f'count={count}')",
        )
        nb_runner.run_cells([2, 3])
        assert "vals=['BIRD', 'CAT', 'DOG']" in nb_runner.get_output(2)
        assert "count=3" in nb_runner.get_output(2)

    def test_dict_missing_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class CountDict(dict):\n    def __init__(self):\n        super().__init__()\n        self.miss_count = 0\n    def __missing__(self, key):\n        self.miss_count += 1\n        self[key] = 0\n        return 0\nprint('CountDict defined')",
                "cd = CountDict()\ncd['a']\ncd['b']\ncd['a']  # not a miss\nprint(f'misses={cd.miss_count}')\nprint(f'keys={sorted(cd.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "misses=2" in nb_runner.get_output(2)
        assert "keys=['a', 'b']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "misses=2" in nb_runner.get_output(2)


# Interaction test: collections.UserDict custom dictionary.
# Tests UserDict subclass with custom __setitem__,
# __getitem__ override, and cross-cell custom dict behavior.
@pytest.mark.timeout(90)
class TestUserDictCustom:
    """Test UserDict custom dictionary across cells."""

    def test_userdict_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: custom UserDict
                "from collections import UserDict\n\nclass TypedDict(UserDict):\n    def __init__(self, key_type, val_type, *args, **kwargs):\n        self._key_type = key_type\n        self._val_type = val_type\n        super().__init__(*args, **kwargs)\n    def __setitem__(self, key, value):\n        if not isinstance(key, self._key_type):\n            raise TypeError(f'Key must be {self._key_type}')\n        if not isinstance(value, self._val_type):\n            raise TypeError(f'Value must be {self._val_type}')\n        super().__setitem__(key, value)\n\ntd = TypedDict(str, int)\ntd['a'] = 1\ntd['b'] = 2\ntd['c'] = 3\nprint(f'data={dict(td)}')",
                # Cell 2: read from typed dict
                "total = sum(td.values())\nkeys = sorted(td.keys())\nprint(f'total={total}')\nprint(f'keys={keys}')",
                # Cell 3: verify type enforcement
                "try:\n    td[42] = 'bad'\n    print('error=none')\nexcept TypeError:\n    print('error=caught_key_type')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'a': 1" in out1
        out2 = nb_runner.get_output(2)
        assert "total=6" in out2
        assert "keys=['a', 'b', 'c']" in out2
        out3 = nb_runner.get_output(3)
        assert "error=caught_key_type" in out3

    def test_userdict_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import UserDict\nclass CaseInsensitiveDict(UserDict):\n    def __setitem__(self, key, value):\n        super().__setitem__(key.lower(), value)\n    def __getitem__(self, key):\n        return super().__getitem__(key.lower())\n\ncid = CaseInsensitiveDict()\ncid['Hello'] = 1\ncid['WORLD'] = 2\nprint(f'keys={sorted(cid.keys())}')",
                "val = cid['hello'] + cid['World']\nprint(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['hello', 'world']" in nb_runner.get_output(1)
        assert "val=3" in nb_runner.get_output(2)

        # Edit to add more items
        nb_runner.set_cell_source(
            1,
            "from collections import UserDict\nclass CaseInsensitiveDict(UserDict):\n    def __setitem__(self, key, value):\n        super().__setitem__(key.lower(), value)\n    def __getitem__(self, key):\n        return super().__getitem__(key.lower())\n\ncid = CaseInsensitiveDict()\ncid['Hello'] = 10\ncid['WORLD'] = 20\ncid['Python'] = 30\nprint(f'keys={sorted(cid.keys())}')",
        )
        nb_runner.run_cells([1, 2])
        assert "val=30" in nb_runner.get_output(2)

    def test_userdict_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import UserDict\nclass DefaultDict(UserDict):\n    def __missing__(self, key):\n        return 0\n\ndd = DefaultDict({'x': 10, 'y': 20})\nresult = dd['x'] + dd['z']  # z returns 0\nprint(f'result={result}')",
                "is_ten = result == 10\nprint(f'is_ten={is_ten}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(1)
        assert "is_ten=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_ten=True" in nb_runner.get_output(2)


# Collections module interaction tests (Counter, OrderedDict, defaultdict, namedtuple).
# Tests that editing data fed into collections types properly invalidates downstream.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestCollectionsInteraction:
    """Test collections module patterns with cache invalidation."""

    def test_counter_edit(self, nb_runner):
        """Editing input to Counter should propagate."""
        nb_runner.create_notebook(
            [
                "from collections import Counter\nwords = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']",
                "counts = Counter(words)",
                "top = counts.most_common(1)[0]",
                "print(f'top={top[0]},count={top[1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "top=apple,count=3" in out

        nb_runner.set_cell_source(
            1, "from collections import Counter\nwords = ['banana', 'banana', 'banana', 'cherry', 'apple']"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "top=banana,count=3" in out

    def test_defaultdict_edit(self, nb_runner):
        """Editing data that populates a defaultdict should propagate."""
        nb_runner.create_notebook(
            [
                "from collections import defaultdict\npairs = [('a', 1), ('b', 2), ('a', 3)]",
                "dd = defaultdict(list)\nfor k, v in pairs:\n    dd[k].append(v)",
                "result_a = sorted(dd['a'])\nresult_b = sorted(dd['b'])",
                "print(f'a={result_a},b={result_b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "a=[1, 3]" in out
        assert "b=[2]" in out

        nb_runner.set_cell_source(
            1, "from collections import defaultdict\npairs = [('a', 10), ('b', 20), ('b', 30), ('a', 40)]"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "a=[10, 40]" in out
        assert "b=[20, 30]" in out

    def test_namedtuple_edit(self, nb_runner):
        """Editing namedtuple instances should propagate."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])",
                "p1 = Point(1, 2)\np2 = Point(3, 4)",
                "dist_sq = (p2.x - p1.x)**2 + (p2.y - p1.y)**2",
                "print(f'dist_sq={dist_sq}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "dist_sq=8" in out

        nb_runner.set_cell_source(2, "p1 = Point(0, 0)\np2 = Point(3, 4)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "dist_sq=25" in out


# Collections module interaction tests.
#
# Tests editing cells using collections types like defaultdict,
# Counter, OrderedDict, deque.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestCollectionsEdits:
    """Editing patterns using collections module types."""

    def test_edit_counter_input(self, nb_runner):
        """Edit input to a Counter."""
        nb_runner.create_notebook(
            [
                "from collections import Counter\nwords = 'the cat sat on the mat'.split()",
                "counts = Counter(words)\nmost = counts.most_common(2)\nprint(f'most = {most}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('the', 2)" in nb_runner.get_output(2)

        # Change input text
        nb_runner.set_cell_source(1, "from collections import Counter\nwords = 'a a a b b c'.split()")
        nb_runner.run_all()
        assert "('a', 3)" in nb_runner.get_output(2)

    def test_edit_defaultdict_factory(self, nb_runner):
        """Edit the default factory of a defaultdict."""
        nb_runner.create_notebook(
            [
                "from collections import defaultdict\ndd = defaultdict(int)",
                "dd['x'] += 1\ndd['y'] += 2\nresult = dict(dd)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'x': 1" in out
        assert "'y': 2" in out

        # Change to list factory
        nb_runner.set_cell_source(1, "from collections import defaultdict\ndd = defaultdict(list)")
        nb_runner.set_cell_source(
            2, "dd['x'].append(1)\ndd['y'].append(2)\ndd['y'].append(3)\nresult = dict(dd)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "'x': [1]" in out2
        assert "'y': [2, 3]" in out2

    def test_edit_deque_maxlen(self, nb_runner):
        """Edit deque maxlen parameter."""
        nb_runner.create_notebook(
            [
                "from collections import deque\nd = deque(maxlen=3)",
                "for i in range(5):\n    d.append(i)\nresult = list(d)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(2)

        # Change maxlen
        nb_runner.set_cell_source(1, "from collections import deque\nd = deque(maxlen=5)")
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)


# Mixed-type collection operation interaction tests.
#
# Tests editing operations on collections containing mixed types
# (lists of dicts, dicts of lists, nested structures).
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMixedCollectionEdits:
    """Editing mixed-type collection operations."""

    def test_edit_list_of_dicts(self, nb_runner):
        """Edit operations on a list of dicts."""
        nb_runner.create_notebook(
            [
                "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}]",
                "names = [r['name'] for r in records]\nprint(f'names = {names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['Alice', 'Bob']" in nb_runner.get_output(2)

        # Change to extract scores
        nb_runner.set_cell_source(2, "scores = [r['score'] for r in records]\nprint(f'scores = {scores}')")
        nb_runner.run_all()
        assert "scores = [90, 85]" in nb_runner.get_output(2)

    def test_edit_dict_of_lists(self, nb_runner):
        """Edit operations on a dict of lists."""
        nb_runner.create_notebook(
            [
                "data = {'fruits': ['apple', 'banana'], 'vegs': ['carrot', 'pea']}",
                "all_items = []\nfor k in data:\n    all_items.extend(data[k])\nprint(f'count = {len(all_items)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(
            1,
            "data = {'fruits': ['apple'], 'vegs': ['carrot', 'pea', 'bean'], 'grains': ['rice']}",
        )
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(2)

    def test_edit_nested_structure_access(self, nb_runner):
        """Edit access patterns on deeply nested structures."""
        nb_runner.create_notebook(
            [
                "tree = {'a': {'b': {'c': 42}}}  # nested tree",
                "val = tree['a']['b']['c']\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Change tree
        nb_runner.set_cell_source(1, "tree = {'a': {'b': {'c': 99, 'd': 100}}}  # nested tree v2")
        nb_runner.set_cell_source(2, "val = tree['a']['b']['c'] + tree['a']['b']['d']\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 199" in nb_runner.get_output(2)

    def test_edit_zip_combination(self, nb_runner):
        """Edit zip-based combination of collections."""
        nb_runner.create_notebook(
            [
                "keys = ['x', 'y', 'z']  # zip source keys",
                "vals = [10, 20, 30]  # zip source vals",
                "combined = dict(zip(keys, vals))\nprint(f'combined = {combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'x': 10" in nb_runner.get_output(3)

        # Change values
        nb_runner.set_cell_source(2, "vals = [100, 200, 300]  # zip source vals v2")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'x': 100" in out
        assert "'z': 300" in out
