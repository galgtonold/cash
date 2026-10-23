"""The walk for the notebook code a value holds reads the cells of a
frame's object columns.

A frame keeps its columns in a block manager the walk did not enter. In
``df['p'].apply(upd)`` with ``upd`` calling ``p.absorb(p)``, ``absorb`` (a
method of the user's class, reading ``Z``) ran with nothing of it in the
key: after ``Z`` changed the statement was served. Columns of numbers and
of plain values give what they gave before.
"""

from __future__ import annotations

import pytest

from cash.notebook import held_functions

pd = pytest.importorskip("pandas")

NOTEBOOK: dict = {"__name__": "__main__"}
exec("class O:\n    def absorb(self, o):\n        o.z = Z\n", NOTEBOOK)
O = NOTEBOOK["O"]


def _codes(value) -> set[str]:
    held_functions._MEMO.clear()
    return {code.co_name for code in held_functions.held_notebook_code(value, NOTEBOOK).codes}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(lambda: pd.DataFrame({"p": [O() for _ in range(5)], "q": range(5)}), id="frame"),
        pytest.param(lambda: pd.Series([O() for _ in range(5)]), id="series"),
        pytest.param(lambda: pd.DataFrame({"p": [1, "a", O()]}), id="mixed_column"),
        pytest.param(lambda: [pd.DataFrame({"p": [O()]})], id="frame_in_a_list"),
    ],
)
def test_the_methods_of_objects_in_a_column_are_held(value):
    assert _codes(value()) == {"absorb"}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(lambda: pd.DataFrame({"q": range(5), "f": [0.5] * 5}), id="numbers"),
        pytest.param(lambda: pd.DataFrame({"s": ["a", "b"], "n": [None, 1]}, dtype=object), id="plain_objects"),
        pytest.param(lambda: pd.Series([{"i": 1}, (2, 3)]), id="dicts_and_tuples"),
    ],
)
def test_a_frame_without_the_users_objects_holds_nothing(value):
    found = held_functions.held_notebook_code(value(), NOTEBOOK)
    assert found.codes == () and found.complete
