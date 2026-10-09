"""A cached function and a cached method over records of a class a cell
defines, in a real kernel.

Such records are keyed by their class and fields at C speed
(`canonical_form._records_by_fields`, `_object_by_fields`). The key must
still change when a field does, when the records are shared differently,
and when the cell defining the class runs again with new code -- and hit
when nothing changed, over Run Alls and a restart.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_CLASS = (
    "import cash\n"
    "class Point:\n"
    "    def __init__(self, x, y, name):\n"
    "        self.x, self.y, self.name = x, y, name\n"
    "    def scaled(self):\n"
    "        return self.y * 1\n"
    "class Model:\n"
    "    def __init__(self, rows):\n"
    "        self.data = rows\n"
    "    @cash.cache\n"
    "    def total(self, k):\n"
    "        return sum(r['v'] for r in self.data) * k\n"
    "@cash.cache\n"
    "def total(points):\n"
    "    return sum(p.scaled() for p in points)\n"
)
_DATA = (
    "points = [Point(i, float(i), f'p{i}') for i in range(2000)]\n"
    "m = Model([{'v': i, 'tags': ['a']} for i in range(2000)])\n"
)
_CALLS = (
    "first = (total(points), m.total(2))\n"
    "points[5].y = 1000.0\n"
    "m.data[3]['v'] = 1000\n"
    "edited = (total(points), m.total(2))\n"
    "check = (first, edited, total.cache_info()['misses'], Model.total.cache_info()['misses'])\n"
)


def test_records_of_a_notebook_class_key_by_their_fields(nb_runner):
    nb_runner.create_notebook([_CLASS, _DATA, _CALLS])
    nb_runner.start_kernel()
    plain_first = (sum(float(i) for i in range(2000)), sum(range(2000)) * 2)
    plain_edited = (plain_first[0] - 5.0 + 1000.0, (sum(range(2000)) - 3 + 1000) * 2)
    expected = f"({plain_first}, {plain_edited}, 2, 2)"
    nb_runner.run_all()
    assert nb_runner.peek("check") == expected

    nb_runner.run_all()
    assert nb_runner.peek("first") == str(plain_first)
    assert nb_runner.peek("edited") == str(plain_edited)

    # The class's code changes: the records are keyed apart.
    nb_runner.set_cell_source(1, _CLASS.replace("return self.y * 1", "return self.y * 2"))
    nb_runner.run_all()
    assert nb_runner.peek("first") == str((plain_first[0] * 2, plain_first[1]))
    assert nb_runner.peek("edited") == str((plain_edited[0] * 2, plain_edited[1]))

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert nb_runner.peek("first") == str((plain_first[0] * 2, plain_first[1]))
    assert nb_runner.peek("edited") == str((plain_edited[0] * 2, plain_edited[1]))
