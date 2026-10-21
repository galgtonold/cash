"""A call that reaches something with identity and returns a library object
too big to read, a call handed a big library object, and a call returning
the user's object from a frame's object column, compared with plain Python
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
}


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
