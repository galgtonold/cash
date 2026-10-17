"""A loop over the rows of a frame a call makes is sized by that frame.

``for i, row in df.head(500).iterrows():`` could not be sized: the length was
read only from a name or a plain attribute, so the call's frame counted as of
unknown length and every pass went through the per-statement machinery, 3 s
against 0.02 s plain. The header's own evaluation now records the length of
the frame the call returns, as it does for ``loader.batch``, and the call is
not made a second time to size the loop.
"""

from __future__ import annotations

import ast

import pytest

from cash.notebook.control_structures import single_unit_policy as policy
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")


def _sized(header: str, ns: dict) -> int | None:
    wrapped, texts = policy.sizing_header(header) or (header, ())
    sizes: dict[str, int | None] = {}
    value = eval(wrapped, {**ns, policy.SIZE_PROBE_NAME: policy.size_probe(texts, sizes)})
    return policy.estimated_iterations(ast.parse(header, mode="eval").body, value, ns, sizes)


@pytest.mark.parametrize(
    ("header", "passes"),
    [
        ("df.head(500).iterrows()", 500),
        ("df.sample(300, random_state=0).itertuples()", 300),
        ("enumerate(df.head(50).itertuples())", 50),
    ],
)
def test_the_frame_a_call_makes_sizes_the_loop(header, passes):
    ns = {"df": pd.DataFrame({"a": range(1000)})}
    assert _sized(header, ns) == passes


def test_a_called_frames_columns_are_not_counted_by_its_rows():
    ns = {"df": pd.DataFrame({"a": range(1000), "b": range(1000)})}
    assert _sized("df.head(500).items()", ns) is None


def test_the_loop_runs_as_one_unit_and_calls_once(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(
        cash_magics,
        "import pandas as pd\ncalls = []\n"
        "def rows(n):\n    calls.append(n)\n    return pd.DataFrame({'a': range(n), 'b': [2] * n})\n",
    )
    processor = cash_magics._statement_processor
    seen = []
    real = processor.process_statement

    def counted(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    processor.process_statement = counted
    try:
        run_cash_cell(cash_magics, "total = 0\nfor i, row in rows(500).iterrows():\n    total += row['a'] * row['b']")
    finally:
        processor.process_statement = real
    assert ns["total"] == 2 * sum(range(500))
    assert ns["calls"] == [500]
    assert len(seen) < 50, len(seen)
