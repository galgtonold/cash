"""A call that reaches something with identity and returns a library object
too big to read, a call handed a big library object, and a call returning
the user's object from a frame's object column or one of its categorical's
objects, one returning a Series over its argument's array inside a
namespace, and one copying a column made for it, compared with plain Python
across two runs and a restart. Prints every run's cell output (warnings
included)."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

SETUP = "import cash\n%cash_on\nimport time, argparse\nimport pandas as pd"
CASES = {
    # reaches rect (identity), result fresh and too big: should be served, and quickly
    "reached_fresh_60k": (
        "rect = [1]",
        "def mk():\n    time.sleep(2.0)\n    k = len(rect)\n"
        "    return argparse.Namespace(rows=[{'i': i} for i in range(60000)], k=k)",
        "t = time.time()\nw = mk()\nprint('T', round(time.time() - t, 1), len(w.rows))",
        None,
    ),
    # reaches rect, result too big AND holds rect: must be refused like plain (SAME True 5)
    "reached_shares_60k": (
        "rect = [1]",
        "def mk():\n    time.sleep(2.0)\n    return argparse.Namespace(rows=[{'i': i} for i in range(60000)], r=rect)",
        "t = time.time()\nw = mk()\nprint('T', round(time.time() - t, 1), len(w.rows))",
        "rect.append(5)\nprint('SAME', w.r is rect, w.r[-1])",
    ),
    # big library argument (Namespace of 1M dicts), fresh big result
    "big_lib_arg": (
        "src = argparse.Namespace(r=[{'i': i} for i in range(1_000_000)])",
        "def mk(s):\n    time.sleep(2.0)\n"
        "    return argparse.Namespace(rows=[{'i': i} for i in range(60000)], n=len(s.r))",
        "t = time.time()\nw = mk(src)\nprint('T', round(time.time() - t, 1), len(w.rows))",
        None,
    ),
    # the user's list inside an object column of a DataFrame argument / global
    "obj_in_df_arg": (
        "rect = [1]\ndf = pd.DataFrame({'d': [rect, [2]]})",
        "def mk(d):\n    time.sleep(1.0)\n    return argparse.Namespace(x=d['d'].iloc[0])",
        "w = mk(df)",
        "rect.append(5)\nprint('SAME', w.x is rect, w.x[-1])",
    ),
    "obj_in_df_global": (
        "rect = [1]\ndf = pd.DataFrame({'d': [rect, [2]]})",
        "def mk():\n    time.sleep(1.0)\n    return argparse.Namespace(x=df['d'].iloc[0])",
        "w = mk()",
        "rect.append(5)\nprint('SAME', w.x is rect, w.x[-1])",
    ),
    "obj_in_df_arg_dict": (
        "rect = [1]\ndf = pd.DataFrame({'d': [rect, [2]]})",
        "def mk(d):\n    time.sleep(1.0)\n    return {'x': d['d'].iloc[0]}",
        "w = mk(df)",
        "rect.append(5)\nprint('SAME', w['x'] is rect, w['x'][-1])",
    ),
    # a Series over the argument's own array, inside a library object: a write to the array shows in it
    "series_on_arg_in_namespace": (
        "import numpy as np\narr = np.arange(5.0)",
        "def mk(a):\n    time.sleep(1.0)\n    return argparse.Namespace(s=pd.Series(a, copy=False))",
        "w = mk(arr)",
        "arr[0] = 5.0\nprint('SAME', w.s.iloc[0] == 5.0, int(w.s.iloc[0]))",
    ),
    # a filter of the argument's frame holds its very lists: served, still holding them
    "frame_filter_of_arg": (
        "df = pd.DataFrame({'t': [['a', str(i)] for i in range(1000)], 'n': range(1000)})",
        "def mk(d):\n    time.sleep(1.0)\n    return d[d['n'] % 2 == 0]",
        "t = time.time()\nw = mk(df)\nprint('T', round(time.time() - t, 1), len(w))",
        "df['t'].iloc[2].append(5)\nprint('SAME', w['t'].iloc[1] is df['t'].iloc[2], w['t'].iloc[1][-1])",
    ),
    "series_filter_of_arg": (
        "df = pd.DataFrame({'t': [['a', str(i)] for i in range(1000)], 'n': range(1000)})",
        "def mk(s):\n    time.sleep(1.0)\n    return s[s.map(len) == 2].iloc[::2]",
        "t = time.time()\nw = mk(df['t'])\nprint('T', round(time.time() - t, 1), len(w))",
        "df['t'].iloc[2].append(5)\nprint('SAME', w.iloc[1] is df['t'].iloc[2], w.iloc[1][-1])",
    ),
    # a copy of a column made for the call: no name holds it for the statement
    "copy_of_a_column": (
        "df = pd.DataFrame({'t': [['a', str(i)] for i in range(1000)], 'n': range(1000)})",
        "def mk(s):\n    time.sleep(1.0)\n    return s.copy()",
        "t = time.time()\nw = mk(df['t'])\nprint('T', round(time.time() - t, 1), len(w))",
        "df['t'].iloc[2].append(5)\nprint('SAME', w.iloc[2] is df['t'].iloc[2], w.iloc[2][-1])",
    ),
    # one of a categorical's objects, held outside a frame
    "category_object_of_arg": (
        "class K:\n    def __init__(self, v):\n        self.v = [v]\n"
        "df = pd.DataFrame({'c': pd.Categorical([K(i) for i in range(3)] * 10), 'n': range(30)})",
        "def pick(d):\n    time.sleep(1.0)\n    return {'first': d['c'].iloc[1]}",
        "t = time.time()\nw = [pick(df)]\nprint('T', round(time.time() - t, 1), len(w))",
        "df['c'].iloc[1].v.append(5)\nprint('SAME', w[0]['first'] is df['c'].iloc[1], w[0]['first'].v[-1])",
    ),
    "category_object_of_arg_statement": (
        "class K:\n    def __init__(self, v):\n        self.v = [v]\n"
        "df = pd.DataFrame({'c': pd.Categorical([K(i) for i in range(3)] * 10), 'n': range(30)})",
        "def pick(d):\n    time.sleep(1.0)\n    return {'first': d['c'].iloc[1]}",
        "t = time.time()\nw = pick(df)\nprint('T', round(time.time() - t, 1), len(w))",
        "df['c'].iloc[1].v.append(5)\nprint('SAME', w['first'] is df['c'].iloc[1], w['first'].v[-1])",
    ),
}

#: Served after the first run.
SERVED = {"frame_filter_of_arg", "series_filter_of_arg", "copy_of_a_column"}


@pytest.mark.parametrize("case", list(CASES))
def test_case(nb_runner, case):
    data, fn, call, check = CASES[case]
    cells = [SETUP, data, fn, call] + ([check] if check else [])
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    seen = []
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        seen.append([nb_runner.get_raw_output(i).strip() for i in range(4, len(cells) + 1)])
    print(case, "runs:", seen)
    if check:
        assert all(s[-1] == "SAME True 5" for s in seen), seen
    else:
        assert all(float(s[0].split()[1]) < 1.5 for s in seen[1:]), seen
    if case in SERVED:
        assert all(float(s[0].split()[1]) < 0.5 for s in seen[1:]), seen
