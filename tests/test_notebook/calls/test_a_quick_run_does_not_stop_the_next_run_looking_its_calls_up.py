"""A quick run of a statement does not stop a run with other inputs looking its calls up.

A statement that ran under the call cost floor with no call served is not
rewritten next time: its calls cannot be worth a key. That mark was kept under
the statement's text, which is the same when a parameter changes and across a
loop's iterations, so ``run(cfg)`` with ``cfg`` set to a quick value and then
back recomputed the slow result the call cache held, and in a loop whose items
alternate quick and slow each slow item after a quick one ran again on every
re-run. The mark is kept per run with the same inputs.
"""

import ast

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _defs(log) -> str:
    return (
        "import os, time\n"
        "def run(cfg):\n"
        f"    fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "    time.sleep(cfg['secs'])\n"
        "    return cfg['secs'] * 10"
    )


def _runs(log) -> int:
    return len(log.read_bytes()) if log.exists() else 0


def test_switching_a_parameter_back_serves_the_stored_call(cash_magics, mock_shell, cash_instance, tmp_path):
    # A floor a loaded machine cannot stretch the quick run past.
    cash_instance.config.call_cost_floor_seconds = 0.05
    log = tmp_path / "runs.log"
    run_cash_cell(cash_magics, _defs(log))
    for secs in (ABOVE_PERSISTENCE_FLOOR_S, 0.0):
        run_cash_cell(cash_magics, f"cfg = {{'secs': {secs}}}")
        run_cash_cell(cash_magics, "res = run(cfg) * 2")
    assert _runs(log) == 2
    run_cash_cell(cash_magics, f"cfg = {{'secs': {ABOVE_PERSISTENCE_FLOOR_S}}}")
    run_cash_cell(cash_magics, "res = run(cfg) * 2")

    assert mock_shell.user_ns["res"] == ABOVE_PERSISTENCE_FLOOR_S * 20
    assert _runs(log) == 2, "run(cfg) was computed again although the call cache held its result"


def test_the_mark_is_kept_per_run_with_the_same_inputs(cash_magics, statement_processor):
    """A quick run spares the rewrite of a run with the same key (the
    statement and its inputs' lineage), and only that one."""
    run_cash_cell(cash_magics, "def bump(v):\n    return v + 1\nx = 1")
    calls = statement_processor._calls
    code = "y = bump(x)"

    def routes(key):
        return calls.code_and_tree_for_execution(code, ast.parse(code), None, key=key)[0] is not code

    assert routes("stmt:quick")
    calls.learn_call_wrapping(code, 0.0001, [])
    assert not routes("stmt:quick"), "the same quick run was rewritten again"
    assert routes("stmt:other-inputs")
    calls.learn_call_wrapping(code, 1.0, [])
    assert routes("stmt:other-inputs")
