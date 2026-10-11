"""A loop run as one unit evaluates its header once, as Python does.

``for a, b in tqdm(list(zip(actions, next_actions))):`` over 2M pairs: the
header was evaluated to size the loop, then again by the unit that ran it, so
the list was built twice and a second bar drawn (3.4 s plain against 8.2 s).
A header that is a call to a pure builtin producer or a progress bar is now
sized from what it wraps, and evaluated only by the unit.

Any other header is evaluated first, and the unit iterates that value. It used
to run the loop from source and evaluate the header again, so a drained queue
gave it nothing, a mapped function ran twice per item and a property gave it a
second batch. Sizing a header never runs a property either. The integration
twin is ``tests/test_notebook_integration/loops/test_a_long_loop_evaluates_its_header_once.py``.
"""

from __future__ import annotations

import ast
import re

import pytest

from cash.notebook.control_structures import single_unit_policy as policy
from tests._cell_driver import run_cash_cell

# The progress bar is one of the headers, and every case imports it.
pytest.importorskip("tqdm")

COUNTED = (
    "class Counted:\n"
    "    def __init__(self, n):\n        self.n, self.iters = n, 0\n"
    "    def __len__(self):\n        return self.n\n"
    "    def __iter__(self):\n        self.iters += 1\n        return iter(range(self.n))\n"
    "xs, ys = Counted(2000), Counted(2000)\n"
)


@pytest.mark.parametrize(
    "header",
    ["list(zip(xs, ys))", "zip(xs, ys)", "sorted(zip(xs, ys))", "tqdm(list(zip(xs, ys)))"],
)
def test_the_header_is_iterated_once(cash_magics, header):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, COUNTED + "from tqdm import tqdm")
    run_cash_cell(cash_magics, f"total = 0\nfor a, b in {header}:\n    total += a * b")
    assert ns["total"] == sum(i * i for i in range(2000))
    assert (ns["xs"].iters, ns["ys"].iters) == (1, 1)


def test_one_bar_is_drawn(cash_magics, capsys):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "from tqdm import tqdm\nxs = list(range(2000))")
    capsys.readouterr()
    run_cash_cell(cash_magics, "total = 0\nfor a, b in tqdm(list(zip(xs, xs))):\n    total += a * b")
    assert ns["total"] == sum(i * i for i in range(2000))
    drawn = capsys.readouterr().err
    assert len(re.findall(r"(?<![0-9])0/2000", drawn)) == 1, drawn  # each bar starts at 0/2000


def test_a_rerun_after_an_edit_reads_the_new_data(cash_magics):
    ns = cash_magics.shell.user_ns
    loop = "total = 0\nfor a, b in list(zip(xs, ys)):\n    total += a * b"
    run_cash_cell(cash_magics, "xs = list(range(2000))\nys = [1] * 2000")
    run_cash_cell(cash_magics, loop)
    assert ns["total"] == sum(range(2000))
    run_cash_cell(cash_magics, "ys = [2] * 2000")
    run_cash_cell(cash_magics, loop)
    assert ns["total"] == 2 * sum(range(2000))


def test_a_shadowed_builtin_is_evaluated_first(cash_magics):
    """A notebook's own ``list`` gets no benefit of the doubt: the header is
    evaluated before the loop is judged, as before."""
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "calls = []\ndef list(x):\n    calls.append(1)\n    return [*x]\nxs = range(2000)")
    run_cash_cell(cash_magics, "total = 0\nfor a in list(xs):\n    total += a")
    assert ns["total"] == sum(range(2000))
    assert len(ns["calls"]) == 1


INBOX = (
    "class Inbox:\n    def __init__(self): self.items = list(range(300))\n"
    "    def drain(self):\n        out, self.items = self.items, []\n        return out\ninbox = Inbox()"
)
LOADER = (
    "class Loader:\n    def __init__(self):\n        self.epoch = 0\n    @property\n    def batch(self):\n"
    "        self.epoch += 1\n        return list(range(self.epoch * 1000, self.epoch * 1000 + 300))\n"
    "loader = Loader()"
)


def test_a_drained_queue_is_drained_once(cash_magics, mock_shell):
    run_cash_cell(cash_magics, INBOX)
    run_cash_cell(cash_magics, "total = 0\nfor v in sorted(inbox.drain()):\n    total += v")
    assert mock_shell.user_ns["total"] == sum(range(300))


def test_a_function_mapped_over_a_range_runs_once_per_item(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "calls = []\ndef f(x):\n    calls.append(x)\n    return x * 2")
    run_cash_cell(cash_magics, "tot = 0\nfor v in list(map(f, range(300))):\n    tot += v")
    assert len(mock_shell.user_ns["calls"]) == 300
    assert mock_shell.user_ns["tot"] == 2 * sum(range(300))


def test_a_property_in_the_header_is_read_once(cash_magics, mock_shell):
    run_cash_cell(cash_magics, LOADER)
    run_cash_cell(cash_magics, "tot = 0\nfor i, x in enumerate(loader.batch):\n    tot += x")
    assert mock_shell.user_ns["loader"].epoch == 1
    assert mock_shell.user_ns["tot"] == sum(range(1000, 1300))
    assert not any(name.startswith("__cash_") for name in mock_shell.user_ns), "a name cash bound was left behind"


def test_sizing_a_header_runs_no_property():
    reads = []

    class Loader:
        @property
        def batch(self):
            reads.append(1)
            return [1, 2, 3]

    node = ast.parse("enumerate(loader.batch)", mode="eval").body
    assert policy.estimated_iterations(node, policy.UNEVALUATED, {"loader": Loader()}) is None
    assert reads == []


def test_plain_data_on_an_object_is_still_sized():
    class Cfg:
        pass

    cfg = Cfg()
    cfg.rows = list(range(300))
    for header, ns in (
        ("enumerate(cfg.rows)", {"cfg": cfg}),
        ("enumerate(table['rows'])", {"table": {"rows": list(range(300))}}),
    ):
        node = ast.parse(header, mode="eval").body
        assert policy.estimated_iterations(node, policy.UNEVALUATED, ns) == 300, header


def test_a_range_of_plain_ints_is_sized_from_its_text():
    for header, expected in (("range(300)", 300), ("range(n)", 300), ("range(0, n, 3)", 100), ("range(10, 0, -1)", 10)):
        node = ast.parse(f"list(map(f, {header}))", mode="eval").body
        assert policy.estimated_iterations(node, policy.UNEVALUATED, {"n": 300, "f": str}) == expected, header


def test_the_header_evaluation_records_the_lengths_it_reads():
    """`a.var["symbol"].items()` is an iterator with no length: it is sized by
    the length of `a.var["symbol"]` as the evaluation read it, not by reading
    the property again."""
    sizing = policy.sizing_header("a.var['symbol'].items()")
    assert sizing is not None
    code, texts = sizing
    reads = []

    class A:
        @property
        def var(self):
            reads.append(1)
            return {"symbol": {i: i for i in range(7)}}

    sizes: dict = {}
    eval(code, {"a": A(), policy.SIZE_PROBE_NAME: policy.size_probe(texts, sizes)})
    node = ast.parse("a.var['symbol'].items()", mode="eval").body
    assert policy.estimated_iterations(node, iter(()), {"a": A()}, sizes) == 7
    assert reads == [1]
