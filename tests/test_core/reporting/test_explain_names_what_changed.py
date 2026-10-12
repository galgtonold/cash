"""After an edit, explain() and cache_info() name what changed, in a fresh
process too, before any call; and when the edit is in code the function
calls, explain() says how to keep the next such edit from rerunning it.

A new process has no ledger of its own for the state it computes until a
call builds one: explain() before the first call said only "code or state
changed", and cache_info() counted the miss under that kind alone. Only the
CASH_DEBUG line named the helper.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

_TICKS = """
def _parse(line):
    ts, price = line.split(",")
    return int(ts), float({price})

def load_ticks(n):
    return [_parse(f"{{i}},{{100 + i % 7}}") for i in range(n)]
"""

_JOB = """
import json
import cash
from ticks import load_ticks

FEE_SCALE = {fee_scale}

@cash.cache
def signals(window):
    ticks = load_ticks(50)
    return [p for _, p in ticks[::window]]

@cash.cache
def backtest(window, fee):
    sig = signals(window)
    return sum(sig) - fee * FEE_SCALE * len(sig)

out = {{f.__name__: dict(f.explain(*a).details) for f, a in ((signals, (5,)), (backtest, (5, 0.1)))}}
backtest(5, 0.1)
out["changed"] = {{f.__name__: f.cache_info()["changed"] for f in (signals, backtest)}}
print("OUT", json.dumps(out))
"""


def _write(tmp_path, price="price", fee_scale=1):
    (tmp_path / "ticks.py").write_text(textwrap.dedent(_TICKS.format(price=price)), encoding="utf-8")
    (tmp_path / "job.py").write_text(textwrap.dedent(_JOB.format(fee_scale=fee_scale)), encoding="utf-8")


def _out(tmp_path) -> dict:
    done = run_python("job.py", cwd=tmp_path)
    line = next(line for line in done.stdout.splitlines() if line.startswith("OUT "))
    return json.loads(line[4:])


def test_an_edited_helper_is_named_with_the_callee_to_pass_in(tmp_path):
    _write(tmp_path)
    _out(tmp_path)
    _write(tmp_path, price="price.strip()")
    out = _out(tmp_path)

    sig, bt = out["signals"], out["backtest"]
    assert sig["changed"] == "helper ticks._parse changed", sig
    assert "helper ticks._parse changed" in sig["why"]
    # _parse is called through load_ticks: the body can be handed what
    # load_ticks returns, not what _parse does.
    assert "what load_ticks returns" in sig["hint"], sig["hint"]
    from cash.decorator.explain import PASS_DATA_IN_DOCS

    assert sig["hint"].endswith(f"({PASS_DATA_IN_DOCS}).")
    assert bt["changed"] == "cached function job.signals changed", bt
    assert "what signals returns" in bt["hint"], bt["hint"]
    assert out["changed"] == {
        "signals": {"helper ticks._parse changed": 1},
        "backtest": {"cached function job.signals changed": 1},
    }


def test_a_changed_global_value_is_named_without_the_hint(tmp_path):
    _write(tmp_path)
    _out(tmp_path)
    _write(tmp_path, fee_scale=2)
    bt = _out(tmp_path)["backtest"]
    assert bt["changed"] == "global FEE_SCALE changed", bt
    assert "hint" not in bt, bt


def test_a_chain_that_cannot_be_followed_gives_no_callee_name():
    from cash.decorator.explain import callee_to_pass_in

    old = {"helper m._parse": "a", "global m (read by f)": "x"}
    new = {"helper m._parse": "b", "global m (read by f)": "x"}
    assert callee_to_pass_in(old, new) == ""
    assert callee_to_pass_in(old, old) is None
