"""Records a cell builds come back from the RAM tier as they were built.

The tier keeps big JSON-like data as bytes it reads back in one pass, where
it copied it a container at a time on the store and on every hit. A hit must
still hand back the records as the cell built them: every type, sign and
key order as it was, one list that every record holds still one list, and
none of what a later cell changes in them -- over Run Alls and a restart.
These pass before the change too: they guard it.
"""

import pytest

from tests._nbharness.badge import shows_cached
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.timeout(300)]

N = 20_000

CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import time",
    (
        "records = [{'id': i, 'tags': common, 'meta': {'k': i, 'even': i % 2 == 0, 'neg': -0.0, 'z': 1j, "
        "'raw': b'z', 'big': 2 ** 70}} "
        f"for common in [['x']] for i in range({N}) if i or not time.sleep({ABOVE_PERSISTENCE_FLOOR_S})]"
    ),
    "records[5]['tags'].append('y')\nrecords[7]['meta']['k'] = 'changed'",
    (
        "r = (records[0]['tags'], records[0]['tags'] is records[-1]['tags'], records[7]['meta'], "
        "type(records[2]['meta']['even']).__name__, repr(records[1]['meta']['neg']), len(records))"
    ),
]

EXPECTED = repr(
    (
        ["x", "y"],
        True,
        {"k": "changed", "even": False, "neg": -0.0, "z": 1j, "raw": b"z", "big": 2**70},
        "bool",
        "-0.0",
        N,
    )
)


@pytest.mark.parametrize("imports", ["import time", "import time\nimport numpy as np"], ids=["plain", "numpy-loaded"])
def test_records_restored_from_the_cache_are_the_records_the_cell_built(nb_runner, imports):
    """With numpy loaded the entry also holds numpy's RNG state, and the
    records are kept apart from it."""
    nb_runner.create_notebook([CELLS[0], imports, *CELLS[2:]])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == EXPECTED, "first Run All"
    for label in ("second", "third"):
        nb_runner.run_all()
        raw = nb_runner.get_raw_output(3)
        assert shows_cached(raw), f"{label} Run All: the records cell is a hit\n{raw}"
        assert nb_runner.peek("r") == EXPECTED, f"{label} Run All"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("r") == EXPECTED, "after a restart"


PARSE_CELLS = [
    "import cash\n%cash_on",
    "import json",
    "with open('events.jsonl') as f:\n    records = [json.loads(line) for line in f]",
    "def parse(r):\n    return {'id': r['id'], 'v': r['value'] * 1.1}",
    "clean = [parse(r) for r in records if r['ok']]",
    "n = len(clean)",
]


def test_records_read_inside_a_with_block_survive_a_helper_edit(nb_runner):
    """``with open(...) as f`` leaves ``f`` bound to a closed file beside the
    records. The tier keeps the records as bytes and the closed file as a
    closed file, which no later copy can copy: a hit handed back the tier's
    own entry, and the cell after an edited helper got the bytes' wrapper
    for ``records`` (``'_Marshalled' object is not iterable``)."""
    import json

    n = 10_000
    lines = (json.dumps({"id": i, "value": i * 0.5, "ok": i % 7 != 0, "tags": ["a"]}) for i in range(n))
    (nb_runner.work_dir / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    expected = sum(1 for i in range(n) if i % 7 != 0)
    nb_runner.create_notebook(PARSE_CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("n") == repr(expected), "first Run All"
    nb_runner.run_all()
    assert nb_runner.peek("type(records).__name__") == "'list'", "second Run All: the records cell is a hit"
    nb_runner.set_cell_source(4, PARSE_CELLS[3].replace("1.1", "1.2"))
    nb_runner.run_all()
    assert nb_runner.peek("type(records).__name__") == "'list'", "after the helper edit"
    assert nb_runner.peek("(n, clean[0]['v'])") == repr((expected, 0.5 * 1.2)), "after the helper edit"
