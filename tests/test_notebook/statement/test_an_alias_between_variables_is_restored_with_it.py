"""A statement whose object another notebook variable holds is cached with it.

``b = a`` and then ``a['x'] = ...``: in plain Jupyter ``b`` sees the change,
because it is the same dict. A hit binds copies, so the statement used to run
every time. Now the variables holding the object (directly, or through the
lists, dicts and attributes in them) are stored with the outputs and restored
as one graph: after a hit ``a is b`` again, and a later in-place change shows
through both.

The expensive part is written inline: a call to a notebook function would be
cached on its own, and the statement around it would not be worth storing.
"""

from __future__ import annotations

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = "import time\nclass Model:\n    def __init__(self):\n        self.w = 0"


def _slow(expr: str) -> str:
    return f"(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), {expr})[1]"


#: Doubles ``d['a']``, slowly.
DOUBLE = "d['a'] = " + _slow("d['a'] * 2")


def _run_all(processor, statements):
    return [processor.process_statement(code) for code in statements]


def _ns(processor):
    return processor.shell.user_ns


def test_an_alias_is_restored_as_the_same_object(cash_magics, statement_processor):
    run_cash_cell(cash_magics, SETUP)
    statements = ["a = {'v': 1}", "b = a", f"a['x'] = {_slow('1')}"]

    first = _run_all(statement_processor, statements)[-1]
    assert first["status"] == CacheStatus.COMPUTED and not first["uncacheable_reasons"], first["uncacheable_reasons"]

    hit = _run_all(statement_processor, statements)[-1]
    assert hit["status"] == CacheStatus.RESTORED
    ns = _ns(statement_processor)
    assert ns["a"] is ns["b"]
    assert ns["b"] == {"v": 1, "x": 1}
    statement_processor.process_statement("a['y'] = 2")
    assert ns["b"]["y"] == 2


def test_a_model_in_a_config_dict_stays_the_configs_model(cash_magics, statement_processor):
    run_cash_cell(cash_magics, SETUP)
    statements = ["clf = Model()", "cfg = {'model': clf}", f"clf.w = {_slow('5')}"]

    _run_all(statement_processor, statements)
    hit = _run_all(statement_processor, statements)[-1]

    assert hit["status"] == CacheStatus.RESTORED
    ns = _ns(statement_processor)
    assert ns["cfg"]["model"] is ns["clf"]
    assert ns["cfg"]["model"].w == 5


def test_an_item_taken_out_of_a_list_is_the_lists_item_again(cash_magics, statement_processor):
    run_cash_cell(cash_magics, SETUP)
    statements = ["rows = [{'a': 1.0}, {'a': 3.0}]", "d = rows[0]", DOUBLE]

    _run_all(statement_processor, statements)
    hit = _run_all(statement_processor, statements)[-1]

    assert hit["status"] == CacheStatus.RESTORED
    ns = _ns(statement_processor)
    assert ns["rows"][0] is ns["d"]
    assert ns["rows"] == [{"a": 2.0}, {"a": 3.0}]


def test_the_same_change_twice_is_applied_twice(cash_magics, statement_processor):
    """The second ``d = rows[0]`` must not hit the first change's entry, which
    would hand back ``rows`` as it was after one doubling: the first change
    moves ``rows``'s lineage on, so the second one is keyed apart."""
    run_cash_cell(cash_magics, SETUP)
    statements = ["rows = [{'a': 1.0}, {'a': 3.0}]", "d = rows[0]", DOUBLE, "d = rows[0]", DOUBLE]

    for _ in range(3):
        metrics = _run_all(statement_processor, statements)
        assert _ns(statement_processor)["rows"] == [{"a": 4.0}, {"a": 3.0}]
    assert [m["status"] for m in (metrics[2], metrics[4])] == [CacheStatus.RESTORED] * 2


def test_a_holder_made_after_the_entry_runs_the_statement_again(cash_magics, statement_processor):
    """``c = a`` did not exist when the entry was stored: restoring copies of
    ``a`` and ``b`` would leave ``c`` without the change."""
    run_cash_cell(cash_magics, SETUP)
    statements = ["a = {'v': 1}", "b = a", f"a['x'] = {_slow('1')}"]
    _run_all(statement_processor, statements)

    again = _run_all(statement_processor, [*statements[:2], "c = a", statements[2]])[-1]

    assert again["status"] == CacheStatus.COMPUTED
    ns = _ns(statement_processor)
    assert ns["c"] is ns["a"] is ns["b"]
    assert ns["c"]["x"] == 1
    hit = _run_all(statement_processor, [*statements[:2], "c = a", statements[2]])[-1]
    assert hit["status"] == CacheStatus.RESTORED
    assert ns["c"] is ns["a"] is ns["b"]


def test_run_all_keeps_the_aliases_the_upstream_check_sees(cash_magics, statement_processor):
    """Cell by cell, with the upstream check simulating the cells above each
    one: the lineages the hit gives the variables stored with an entry are the
    ones the simulation gives them, so nothing is re-run to "repair" them and
    split the objects apart again."""
    cells = [
        SETUP,
        "a = {'v': 1}\nb = a",
        f"a['x'] = {_slow('1')}",
        "clf = Model()\ncfg = {'model': clf}",
        f"clf.w = {_slow('5')}",
        "b['y'] = 2\ncfg['model'].w += 1\nsame = (a is b, a.get('y'), cfg['model'] is clf, clf.w)",
    ]
    for _ in range(3):
        statuses = []
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
            statuses.append([m.get("status") for m in cash_magics.cash_status("dict")["last_cell"]["statements"]])
        assert _ns(statement_processor)["same"] == (True, 2, True, 6)
    assert statuses[2] == statuses[4] == [CacheStatus.RESTORED]
