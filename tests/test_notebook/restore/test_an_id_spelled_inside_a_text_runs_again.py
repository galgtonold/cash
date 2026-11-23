"""A value holding an object's ``id()`` spelled inside a longer text runs
again, as one holding the bare number does.

``key = 'k' + str(id(sales))``, ``f"run-{id(x)}"``, ``'k' + hex(id(x))`` or
an object's default ``repr`` (``<Q object at 0x7f...>``) carry the address
of an object the notebook holds. The identity check read a text only when it
was a number as a whole (``str(id(x))``, ``hex(id(x))``), so such a value
was stored and, after a restart, served with the old process's address,
where plain Python spells the new object's. Each holder is read by another
path of the check: a single text, a short list, a big flat list, plain
records read at C speed, sessions of records, and pandas columns of Python
and of Arrow text.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import forget_flat_reads, identity_held_reason
from tests._cell_driver import run_cash_cell

MAKE = (
    "class Q:\n    pass\nclass P:\n    def __init__(self):\n        self.seen = []\n        self.child = Q()\np = P()"
)

HOLDERS = {
    "a prefixed decimal id": ("key = 'k' + str(id(p.child))", "r = key == 'k' + str(id(p.child))"),
    "an id in an f-string": ("key = f'run-{id(p.child)}-x'", "r = key == f'run-{id(p.child)}-x'"),
    "a prefixed hex id": ("key = 'k' + hex(id(p.child))", "r = key == 'k' + hex(id(p.child))"),
    "an object's repr": ("key = repr(p.child)", "r = key == repr(p.child)"),
    "two ids run together": (
        "key = str(id(p.seen)) + str(id(p.child))",
        "r = key == str(id(p.seen)) + str(id(p.child))",
    ),
    "an id in a text of a short list": (
        "keys = ['a'] * 9 + [f'obj={id(p.child)}']",
        "r = keys[-1] == f'obj={id(p.child)}'",
    ),
    "an id in a text of a big list": (
        "keys = [f'k{i}' for i in range(20_000)] + [f'obj={id(p.child)};']",
        "r = keys[-1] == f'obj={id(p.child)};'",
    ),
    "an id in a long text with a line break": (
        "key = 'header line\\n' + 'x' * 40 + f' at {hex(id(p.child))} end'",
        "r = key == 'header line\\n' + 'x' * 40 + f' at {hex(id(p.child))} end'",
    ),
    "an id in a text of plain records": (
        "reg = {'a': [{'k': f'id={id(p.child)}'}], 'b': list(range(5))}",
        "r = reg['a'][0]['k'] == f'id={id(p.child)}'",
    ),
    "an id in a text of a list of sessions": (
        "sessions = [[(f'a{j}', j) for j in range(20)] for _ in range(300)] + [[(f'ref:{id(p.child)}', 0)]]",
        "r = sessions[-1][-1][0] == f'ref:{id(p.child)}'",
    ),
    "an id in a pandas column of text": (
        "import pandas as pd\ndf = pd.DataFrame({'key': ['x', f'run-{id(p.child)}'], 'v': ['a', 'b']}, dtype=object)",
        "r = df['key'][1] == f'run-{id(p.child)}'",
    ),
    "an id in a short Arrow text column": (
        "import pandas as pd\n"
        "df = pd.DataFrame({'key': pd.array(['x', f'run-{id(p.child)}'], dtype='string[pyarrow]'), 'v': [1, 2]})",
        "r = df['key'][1] == f'run-{id(p.child)}'",
    ),
    "an id in a long Arrow text column": (
        "import pandas as pd\n"
        "df = pd.DataFrame({'key': [f'S{i % 97:03d}-{i:05d}' if i != 4321 else f'S-{id(p.child)}' for i in range(5000)]"
        ", 'v': range(5000)}, dtype='string[pyarrow]')",
        "r = df['key'][4321] == f'S-{id(p.child)}'",
    ),
    "a hex id in a long Arrow text column of long texts": (
        "import pandas as pd\n"
        "df = pd.DataFrame({'key': ['a sentence of words longer than an address ' * 2] * 4321"
        " + ['a sentence ' * 5 + repr(p.child)]}, dtype='string[pyarrow]')",
        "r = df['key'][4321] == 'a sentence ' * 5 + repr(p.child)",
    ),
}


def _stored_however_cheap(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.mark.parametrize("holder", list(HOLDERS))
def test_a_run_all_spells_the_new_object(cash_magics, cash_instance, holder):
    if "pandas" in holder or "Arrow" in holder:
        pytest.importorskip("pandas")
    if "Arrow" in holder:
        pytest.importorskip("pyarrow")
    _stored_however_cheap(cash_instance)
    hold, read = HOLDERS[holder]
    cells = [MAKE, hold, read + "  # @cash:no-cache"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] is True


@pytest.mark.parametrize(
    "value",
    [
        "key = 'run-' + str(len(p.seen) + 1234567)",
        "keys = [f'S{i % 97:03d}-{i:05d}' for i in range(20_000)]",
        "stamps = [f'at {1_700_000_000 + i}' for i in range(50)]",
        "words = ['order 0x12 shipped', 'a long text ' * 20, 'line\\nbreak']",
    ],
)
def test_a_text_spelling_no_object_s_id_is_restored(cash_magics, cash_instance, value):
    """Positive control: text with numbers that are no live object's id()
    is stored and restored as before."""
    _stored_however_cheap(cash_instance)
    cells = [MAKE, value]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert "RESTORED" in statuses[-1], statuses


@pytest.mark.parametrize(
    ("text", "spelled"),
    [
        ("k{d}", "d"),
        ("run-{d}-x", "d"),
        ("k{h}", "h"),
        ("<Q object at {h}>", "h"),
        ("x{h}zz", "h"),
        ("123456{d}", "d"),  # glued to other digits: a run longer than any address
        ("{d}{d}", "d"),
        ("a\n{d}\nb", "d"),
    ],
)
def test_the_check_finds_the_spelling_anywhere_in_a_text(text, spelled):
    held = object()
    value = text.format(d=str(id(held)), h=hex(id(held)))
    forget_flat_reads()
    for holder in (value, [value], ["a"] * 20_000 + [value], {"k": [value]}):
        reason = identity_held_reason("v", holder, {"v": holder, "held": held})
        assert reason is not None and "id()" in str(reason), (holder if holder is value else type(holder), spelled)


def test_a_number_too_short_for_an_address_is_not_read():
    """Seven digits is the shortest spelling read (`_LOWEST_ADDRESS` has
    seven): a six-digit run is no address, inside a text or alone."""
    found = identity_holders._spelled_in("a123456b c1234567d 0X1234567")
    assert {1234567, 0x1234567} <= found and 123456 not in found


@pytest.mark.parametrize(
    "text",
    ["v2{d}", "{d}7", "9{d}1", "{h}ff", "<{h}abc0>"],
)
def test_an_address_glued_to_other_digits_is_read(text):
    """An address of 14 decimal digits (a heap below the 15-digit mmap
    range) glued to other digits is a run of at most 15, read only as one
    number before; hexadecimal digits glued after ``hex(id(x))`` made a
    run read as one number too. Read inside `reading_every_spelling`, as
    the outputs of a statement reading ``id`` are."""
    address = 94_123_456_789_008
    assert address % 8 == 0 and len(str(address)) == 14
    with identity_holders.reading_every_spelling():
        assert address in identity_holders._spelled_in(text.format(d=address, h=hex(address)))


def test_a_hash_after_0x_is_not_read_by_its_starts():
    """``0x`` and 64 hexadecimal digits is a hash, not an address with
    digits glued after it: none of its starts is read."""
    with identity_holders.reading_every_spelling():
        assert identity_holders._spelled_in("0x" + "7f" * 32) == set()
