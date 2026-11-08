"""A hit writes back every argument the call changed in place, not only
the plain lists among them.

A plain list is checked by the identities of what it holds, which is cheap
and names it at once. That check speaks for the lists alone: a dict or an
array changed beside it was left out of what the entry stored, so a hit
sorted the caller's list and left the dict and the array as they were.
"""

from __future__ import annotations

import textwrap

import numpy as np

from tests._scripts import run_python


def test_a_list_a_dict_and_an_array_are_all_written_back(disk_cash):
    runs = []

    @disk_cash.cache
    def f(a, d, arr):
        runs.append(1)
        a.sort()
        d["n"] = 1
        arr += 1
        return 0

    for _ in range(2):
        a, d, arr = [3, 1, 2], {"k": 1}, np.zeros(2)
        f(a, d, arr)
        assert (a, d, arr.tolist()) == ([1, 2, 3], {"k": 1, "n": 1}, [1.0, 1.0])
    assert len(runs) == 1

    # Control: an argument whose hash before the body says it did not
    # change does not stop the store.
    @disk_cash.cache
    def g(a, arr):
        runs.append(1)
        a.sort()
        return float(arr.sum())

    for _ in range(2):
        a = [3, 1, 2]
        g(a, np.zeros(2))
        assert a == [1, 2, 3]
    assert len(runs) == 2


SCRIPT = textwrap.dedent("""
    import cash

    @cash.cache
    def f(a, d):
        a.sort()
        d["n"] = len(d)
        return len(a)

    a, d = [3, 1, 2], {"k": 1}
    f(a, d)
    print(a, d, f.cache_info()["hits"])
""")


def test_a_new_process_writes_back_both(tmp_path):
    (tmp_path / "job.py").write_text(SCRIPT, encoding="utf-8")
    first = run_python("job.py", cwd=tmp_path).stdout.strip()
    second = run_python("job.py", cwd=tmp_path).stdout.strip()
    assert first == "[1, 2, 3] {'k': 1, 'n': 1} 0"
    assert second == "[1, 2, 3] {'k': 1, 'n': 1} 1"
