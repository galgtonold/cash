"""After ``for r in records:`` the loop variable is still bound to the last
record, and ``for (name, act) in parsed:`` leaves ``act`` bound to a list
inside ``parsed``. Every later statement whose outputs hold those records is
checked for the other names holding part of them; the check reads the
records a level at a time, and must still keep the names bound to the very
objects inside the records, as plain Python does, over Run Alls and a
restart.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]


def test_a_record_the_loop_variable_holds_is_still_the_records(nb_runner):
    nb_runner.create_notebook(
        [
            "records = [{'id': i, 'tags': ['a']} for i in range(5000)]",
            "for r in records:\n    r['s'] = r['id'] * 2",
            "r['tags'].append('last')",
            "check = (records[-1]['tags'], r is records[-1], records[10]['s'])",
        ]
    )
    nb_runner.start_kernel()
    expected = "(['a', 'last'], True, 20)"
    for label in ("first", "second"):
        nb_runner.run_all()
        assert nb_runner.peek("check") == expected, f"{label} Run All"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("check") == expected, "after a restart"


def test_re_sorted_pairs_keep_the_list_a_loop_variable_holds(nb_runner):
    nb_runner.create_notebook(
        [
            "parsed = [(f'u{i}', [('a', i), ('b', -i)]) for i in range(5000)]",
            "for (name, act) in parsed:\n    if len(act) == 0:\n        print(name)",
            "parsed = sorted(parsed, key=lambda x: x[1][1][1])",
            "act.append(('c', 0))",
            "check = (parsed[0][0], any(p[1] is act for p in parsed), [p for p in parsed if len(p[1]) == 3][0][0])",
        ]
    )
    nb_runner.start_kernel()
    expected = "('u4999', True, 'u4999')"
    for label in ("first", "second"):
        nb_runner.run_all()
        assert nb_runner.peek("check") == expected, f"{label} Run All"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("check") == expected, "after a restart"
