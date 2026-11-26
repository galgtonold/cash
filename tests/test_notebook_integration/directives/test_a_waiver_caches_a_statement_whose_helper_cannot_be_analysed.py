"""``# @cash:assume-safe`` caches a statement whose helper cash cannot analyse.

With the analysis of ``fresh_token`` raising, cash cannot tell that it
returns a fresh UUID, and a statement calling it runs every time. The waiver
accepts what the helper reads: the waived statement is served its first
value on the next Run All, the one without a waiver still runs, and an edit
to the helper runs both again, as its code is in the key all along.
"""

import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

PATCH = """
from cash.notebook import lineage_formula

_real = lineage_formula.get_analyzer


class _Unreadable:
    def analyze(self, fn):
        if getattr(fn, "__name__", "") == "fresh_token":
            raise RuntimeError("a construct the analysis cannot read")
        return _real().analyze(fn)


def on():
    lineage_formula.get_analyzer = _Unreadable
    return "on"


def off():
    lineage_formula.get_analyzer = _real
    return "off"
"""

HELPER = "import time\nimport uuid\n\ndef fresh_token():\n    time.sleep(0.3)\n    return uuid.uuid4().hex"
CELLS = [HELPER, "t = fresh_token()  # @cash:assume-safe", "u = fresh_token()"]


def test_the_waived_statement_is_served_and_follows_an_edit_to_the_helper(nb_runner):
    patch = f"unreadable_{uuid.uuid4().hex[:8]}"
    (nb_runner.work_dir / f"{patch}.py").write_text(PATCH, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    try:
        assert nb_runner.peek(f"__import__({patch!r}).on()") == "'on'"
        nb_runner.run_all()
        t, u = nb_runner.peek("t"), nb_runner.peek("u")
        assert len(t) == len(u) == 34 and t != u

        nb_runner.run_all()
        assert nb_runner.peek("t") == t, "the waived statement ran again"
        assert nb_runner.peek("u") != u, "without a waiver the fresh id was frozen"
        assert "NOTEBOOK-ANALYSIS-FAILED" not in nb_runner.get_raw_output(2)

        nb_runner.set_cell_source(1, HELPER.replace(".hex", ".hex.upper()"))
        nb_runner.run_all()
        after = nb_runner.peek("t")
        assert after != t and after.isupper(), "served for the helper as it was before the edit"
    finally:
        nb_runner.peek(f"__import__({patch!r}).off()")
