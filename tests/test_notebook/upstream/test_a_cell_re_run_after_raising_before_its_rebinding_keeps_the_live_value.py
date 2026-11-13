"""A cell re-run after an attempt that raised before it rebound a name keeps
the value the name holds, as a plain kernel does.

An accumulator (``acc = acc + new`` in a loop) is run again after a run that
raised part-way, so it continues from what that run left. An attempt that
raises before the loop reaches the rebinding leaves ``acc`` as it found it.
The upstream check took the next run for a re-run from the object that
attempt started from, read ``acc`` as the cell's own stale output, and put
back the empty value the cell above makes: the rows were lost, where a plain
kernel keeps them.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

_KINDS = {
    "list": ("acc = []", "for i in range(n):\n    new = [i * {f}]\n    acc = acc + new"),
    "frame": (
        "import pandas as pd\nacc = pd.DataFrame({{'a': []}})",
        "for i in range(n):\n    new = pd.DataFrame({{'a': [i * {f}]}})\n    acc = pd.concat([acc, new], ignore_index=True)",
    ),
}
#: Raises at the last pass, after rebinding; raises at the first pass,
#: before rebinding; runs through.
_PART_WAY, _BEFORE_REBINDING, _THROUGH = "(1 if i < n - 1 else None)", "factor", "1"


def _attempts(run, above: str, body: str, factors: list[str]) -> list[int]:
    sizes = []
    run(above.format(), [above.format(), body.format(f=factors[0])])
    for f in factors:
        cell = body.format(f=f)
        try:
            run(cell, [above.format(), cell])
        except (TypeError, NameError):
            pass
        sizes.append(run.size())
    return sizes


class _Plain:
    def __init__(self, n: int):
        self.ns: dict = {"n": n}

    def __call__(self, cell: str, cells: list[str]) -> None:
        exec(cell, self.ns)

    def size(self) -> int:
        return len(self.ns["acc"])


class _Cash:
    def __init__(self, magics, n: int):
        self.magics = magics
        magics.shell.user_ns["n"] = n

    def __call__(self, cell: str, cells: list[str]) -> None:
        run_cash_cell(self.magics, cell, cells=cells)

    def size(self) -> int:
        return len(self.magics.shell.user_ns["acc"])


@pytest.mark.parametrize("kind", list(_KINDS))
@pytest.mark.parametrize(
    "factors",
    [
        [_PART_WAY, _BEFORE_REBINDING, _THROUGH],
        [_PART_WAY, _BEFORE_REBINDING, _BEFORE_REBINDING + "2", _THROUGH],
    ],
    ids=["one-failed-attempt", "two-failed-attempts"],
)
def test_a_failed_attempt_before_the_rebinding_does_not_reset_the_value(cash_magics, kind, factors):
    above, body = _KINDS[kind]
    # Large enough that the value is followed by its identity, not its content.
    n = 3000 if kind == "list" else 1000
    plain = _attempts(_Plain(n), above, body, factors)
    assert plain[-1] == 2 * n - 1
    assert _attempts(_Cash(cash_magics, n), above, body, factors) == plain


def test_a_rebinding_to_the_object_a_helper_changed_in_place_is_still_reset(cash_magics):
    """The control: ``acc = grow(acc)`` binds the very object ``grow``
    changed in place, then the cell raises. The run did write to the
    object it started from, so the next run starts again from what the
    cells above make, as Restart & Run All does."""
    ns = cash_magics.shell.user_ns
    cells = [
        "def grow(rows):\n    rows.append([len(rows)])\n    return rows",
        "acc = [[i] for i in range(3000)]",
        "acc = grow(acc)\nchecked = 1 / flag",
    ]
    ns["flag"] = 1
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns["flag"] = 0
    with pytest.raises(ZeroDivisionError):
        run_cash_cell(cash_magics, cells[2], cells=cells)
    assert len(ns["acc"]) == 3001
    ns["flag"] = 1
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert len(ns["acc"]) == 3001
