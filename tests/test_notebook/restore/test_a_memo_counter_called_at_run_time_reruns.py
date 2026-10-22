"""A statement whose run called a memo's ``cache_info`` or ``cache_clear``,
by a name no text holds (``'_'.join(['cache', 'info'])``) or at a site that
called something else first (a loop over ``[len, f.cache_info]``), is not
served with the memo filled again: its trace saw the counter called. A
statement whose run called no counter is still served."""

from __future__ import annotations

import pytest

from cash.decorator import code_trace
from tests.test_notebook.restore.test_review_memofill4_e2e import _plain
from tests.test_notebook.restore.test_review_memofill5_e2e import SLEEP, _cycle2

pytestmark = pytest.mark.skipif(not code_trace.on(), reason="what ran is recorded on Python 3.12+ only")

IMPORTS = "import functools, operator, time"
MEMO = "@functools.cache\ndef f(n):\n    return n * 2"
INFO = "def info(g, k):\n    return getattr(g, k)()"
EACH = "def each(calls):\n    out = None\n    for g, args in calls:\n        out = g(*args)\n    return out"

# One call site: ``len`` first, the counter next.
STATS = (
    "def stats(m):\n"
    "    out = None\n"
    "    for g, args in ((len, ([],)), (getattr(m, '_'.join(['cache', 'info'])), ())):\n"
    "        out = g(*args)\n"
    "    return out"
)

CASES = {
    "joined_with_a_separator": (INFO, "info(f, '_'.join(['cache', 'info']))"),
    "stripped": (INFO, "info(f, ' cache_info '.strip())"),
    "one_site_after_len": (STATS, "stats(f)"),
    "a_dict_key_handed_on": ("fs = [f]", "getattr(fs[0], next(iter({'cache_info': 0})))()"),
    "methodcaller_of_a_built_name": ("", "operator.methodcaller('_'.join(['cache', 'info']))(f)"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_counter_read_however_named_reruns(cash_magics, case):
    setup, expr = CASES[case]
    statement = f"a = (f(1), {expr}.currsize, time.sleep({SLEEP}))[:2]"
    cells1 = [IMPORTS, MEMO, setup or "pass", "x = f(5)", statement]
    cells2 = [IMPORTS, MEMO, setup or "pass", "x = f(5) + f(6)", statement]
    plain, _ = _plain(cells2)
    _cycle2(cash_magics, cells1, cells2)
    assert cash_magics.shell.user_ns.get("a") == plain["a"] == (2, 3)


def test_a_statement_that_called_no_counter_is_served(cash_magics):
    statement = f"a = (f(1), each([(len, ([],)), (abs, (-3,))]), time.sleep({SLEEP}))[:2]"
    cells = [IMPORTS, MEMO, EACH, "x = f(5)", statement]
    _, seen, _ = _cycle2(cash_magics, cells, cells)
    assert seen[-1] == ["RESTORED"]
    assert cash_magics.shell.user_ns.get("a") == (2, 3)
