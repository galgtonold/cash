"""A loop whose statements call through the call cache measures its passes
by what their code cost, not by cash's time keying the calls.

``action = re.findall(pattern, part)[0]`` takes a few microseconds; the call
routed through the call cache took some 2 ms of cash's own keying and
storing, and that counted as the statement's work. A pass of such statements
then never measured cheap (`passes_were_cheap`), so a 36-pass loop of five
of them ran every pass one by one: 187 statements a run, about 1 s against
0.03 s plain, where the rest goes as one unit after five passes. Counts the
statements run, not the time; the measure itself is wall time, so a pass or
two of noise is allowed, far from the 36 the loop has.
"""

from __future__ import annotations

import pytest

from cash.notebook.control_structures import single_unit_policy
from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell

LOG = "@User9: " + " -> ".join(f"Action_{i % 7} (15/10/19 | 18:{i:02d}:02)" for i in range(36))

LOOP = (
    "rows = []\n"
    "for i in range(0, len(parts)):\n"
    "    action = re.findall(r'[Action_END]\\S*', parts[i])[0]\n"
    "    date = re.findall(r'[(]\\S*', parts[i])[0].replace('(', '')\n"
    "    when = re.findall(r'\\S*[d+:]\\S*', parts[i])[0].replace(')', '')\n"
    "    row = {'action': action, 'date': date, 'time': when}\n"
    "    rows.append(row)"
)

_NOISY_PASSES = 3


@pytest.fixture
def statements(monkeypatch):
    """How many statements went through the statement processor."""
    seen = []
    real = StatementProcessor.process_statement

    def counting(self, *a, **k):
        seen.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(StatementProcessor, "process_statement", counting)
    return seen


def test_cheap_passes_of_routed_calls_hand_the_rest_to_one_unit(cash_magics, statements):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, f"import re\nparts = {LOG!r}.split(':', 1)[1].split('->')")
    statements.clear()
    run_cash_cell(cash_magics, LOOP)
    assert [r["action"] for r in ns["rows"]] == [f"Action_{i % 7}" for i in range(36)]
    assert ns["rows"][-1] == {"action": "Action_0", "date": "15/10/19", "time": "18:35:02"}
    # `rows = []`, five passes of five statements, and the rest's unit: 181
    # statements before, every pass one by one.
    passes = single_unit_policy.CHEAP_PASSES
    assert len(statements) <= 2 + (passes + _NOISY_PASSES) * 5, len(statements)


def test_passes_whose_work_is_most_of_their_time_stay_one_by_one(cash_magics, statements):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, f"import re\nimport time\nparts = {LOG!r}.split(':', 1)[1].split('->')")
    statements.clear()
    run_cash_cell(
        cash_magics,
        "rows = []\nfor i in range(0, len(parts)):\n"
        "    action = re.findall(r'[Action_END]\\S*', parts[i])[0]\n"
        "    rows.append([action, time.sleep(0.02)][0])",
    )
    assert ns["rows"] == [f"Action_{i % 7}" for i in range(36)]
    assert len(statements) >= 1 + 36 * 2, len(statements)
