"""``CASH_VERIFY=1`` turns on every self-check: each memo and fast path then
works its answer out again the slow way behind every reuse and raises when
the two differ.

Each check has a switch of its own, and a suite run with only some of them
on leaves the others' layers unchecked. One switch runs them all; a check
set to ``0`` itself stays off under it, and without either nothing is
checked.
"""

from __future__ import annotations

import json

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

SWITCHES = [
    "CASH_HITPATH_VERIFY",
    "CASH_STATE_MEMO_VERIFY",
    "CASH_CLASS_MEMO_VERIFY",
    "CASH_ANALYSIS_MEMO_VERIFY",
    "CASH_CWATCH_VERIFY",
    "CASH_VERIFY_INCREMENTAL_SIM",
]

SCRIPT = """
import json
from cash import _native, value_watch
from cash.analysis import tree_memo
from cash.decorator import class_memo, hit_memo, state_memo
from cash.notebook.upstream import cell_memo

print(json.dumps({
    "extension": _native.cwatch is not None,
    "CASH_HITPATH_VERIFY": hit_memo.VERIFY,
    "CASH_STATE_MEMO_VERIFY": state_memo.VERIFY,
    "CASH_CLASS_MEMO_VERIFY": class_memo.VERIFY,
    "CASH_ANALYSIS_MEMO_VERIFY": tree_memo.VERIFY,
    "CASH_CWATCH_VERIFY": value_watch.VERIFY,
    "CASH_VERIFY_INCREMENTAL_SIM": cell_memo.verify_requested(),
}))
"""

#: The checks of layers that exist only with the C extension.
NEED_EXTENSION = {"CASH_HITPATH_VERIFY", "CASH_STATE_MEMO_VERIFY", "CASH_CLASS_MEMO_VERIFY"}


def _checks(tmp_path, **env):
    cleared = dict.fromkeys([*SWITCHES, "CASH_VERIFY"])
    out = run_python("-c", SCRIPT, cwd=tmp_path, env={**cleared, **env})
    return json.loads(out.stdout.strip().splitlines()[-1])


def _expected(found, on):
    return {name: name in on and (found["extension"] or name not in NEED_EXTENSION) for name in SWITCHES}


def test_nothing_is_checked_by_default(tmp_path):
    found = _checks(tmp_path)
    assert {name: found[name] for name in SWITCHES} == dict.fromkeys(SWITCHES, False)


def test_the_one_switch_turns_every_check_on(tmp_path):
    found = _checks(tmp_path, CASH_VERIFY="1")
    assert {name: found[name] for name in SWITCHES} == _expected(found, set(SWITCHES))


@pytest.mark.parametrize("off", ["CASH_ANALYSIS_MEMO_VERIFY", "CASH_CWATCH_VERIFY", "CASH_VERIFY_INCREMENTAL_SIM"])
def test_a_check_set_to_0_stays_off_under_it(tmp_path, off):
    found = _checks(tmp_path, CASH_VERIFY="1", **{off: "0"})
    assert {name: found[name] for name in SWITCHES} == _expected(found, set(SWITCHES) - {off})


def test_a_check_of_its_own_still_works_alone(tmp_path):
    found = _checks(tmp_path, CASH_ANALYSIS_MEMO_VERIFY="1")
    assert {name: found[name] for name in SWITCHES} == _expected(found, {"CASH_ANALYSIS_MEMO_VERIFY"})
