"""``a, b = build()`` refers to the call's entry without pickling the result.

Measured on a user's notebook under pandas 2.3:
``n_w, inside_w, _ = net_returns(orders, 12)`` took 7 s where plain Jupyter
took 3. Of cash's 4 s, 2.6 s pickled the 1.7 GiB result for a digest and 0.5 s
copied it into RAM a second time for the statement, because refs matched only
the whole returned tuple, never the names unpacked from it.

When the statement is nothing but ``names = call(...)``, nothing can change
the result between the call's return and the statement's store: the statement
refers to the call's entry, by item for unpacked names, without a digest.
Anything else still proves "unchanged" by digesting.
"""

import pytest

from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.notebook import call_entries, call_refs  # noqa: E402

BUILD = (
    "import time\nimport numpy as np\nimport pandas as pd\n"
    "def build():\n"
    "    time.sleep(0.12)\n"
    "    return pd.DataFrame({'x': np.arange(6_000_000, dtype=float)}), 7\n"
    "def bump(pair):\n"
    "    pair[0].loc[0, 'x'] = -1.0\n"
    "    return pair\n"
)


@pytest.fixture
def seen(monkeypatch):
    """What each statement stored as its variables, and every pickle for a digest."""
    stored, digests = [], []
    real_refs = call_refs.with_call_refs

    def spy_refs(*a, **k):
        out = real_refs(*a, **k)
        stored.append(out)
        return out

    monkeypatch.setattr("cash.notebook.statement.call_routing.with_call_refs", spy_refs)
    real_digest = call_refs.digest_of
    monkeypatch.setattr(call_refs, "digest_of", lambda v: digests.append(1) or real_digest(v))
    real_ds = call_entries.digest_and_size
    monkeypatch.setattr(call_entries, "digest_and_size", lambda v: digests.append(1) or real_ds(v))
    return stored, digests


def test_unpacked_names_refer_to_items_of_the_result(cash_magics, seen):
    stored, digests = seen
    run_cash_cell(cash_magics, BUILD)
    stored.clear()
    run_cash_cell(cash_magics, "a, b = build()")
    ns = cash_magics.shell.user_ns
    assert ns["b"] == 7 and len(ns["a"]) == 6_000_000
    variables = stored[-1]
    assert isinstance(variables["a"], call_refs.CallRef) and variables["a"].item == 0
    assert isinstance(variables["b"], call_refs.CallRef) and variables["b"].item == 1
    assert digests == [], "pickled a 48 MB result that is not worth its bytes"

    # served from the cache: the references resolve to the values
    del ns["a"], ns["b"]
    run_cash_cell(cash_magics, "a, b = build()")
    assert ns["b"] == 7 and float(ns["a"]["x"].iloc[5]) == 5.0


def test_a_result_changed_after_its_call_is_not_referenced(cash_magics, seen):
    stored, _digests = seen
    run_cash_cell(cash_magics, BUILD)
    stored.clear()
    run_cash_cell(cash_magics, "p = bump(build())")
    variables = stored[-1]
    assert not isinstance(variables["p"], call_refs.CallRef)
    assert float(cash_magics.shell.user_ns["p"][0]["x"].iloc[0]) == -1.0


def test_a_subscript_of_the_result_is_not_trusted(cash_magics, seen):
    stored, digests = seen
    run_cash_cell(cash_magics, BUILD)
    stored.clear()
    run_cash_cell(cash_magics, "first = build()[0]")
    assert not isinstance(stored[-1]["first"], call_refs.CallRef)


def test_a_plain_value_worth_keeping_is_not_pickled_either(cash_magics, seen):
    """A real result of 402 MiB for 3.7 s: worth keeping, and the digest the
    statement's trusted reference does not need took 2.6 s."""
    stored, digests = seen
    run_cash_cell(
        cash_magics,
        BUILD + "def small():\n    time.sleep(0.12)\n    return pd.DataFrame({'x': np.arange(1000, dtype=float)})\n",
    )
    stored.clear()
    run_cash_cell(cash_magics, "s = small()")
    assert isinstance(stored[-1]["s"], call_refs.CallRef)
    assert digests == []
    del cash_magics.shell.user_ns["s"]
    run_cash_cell(cash_magics, "s = small()")
    assert float(cash_magics.shell.user_ns["s"]["x"].iloc[999]) == 999.0


def test_a_value_used_elsewhere_is_still_digested(cash_magics, seen):
    stored, digests = seen
    run_cash_cell(
        cash_magics,
        BUILD + "def small():\n    time.sleep(0.12)\n    return pd.DataFrame({'x': np.arange(1000, dtype=float)})\n",
    )
    run_cash_cell(cash_magics, "acc = []")
    run_cash_cell(cash_magics, "acc.append(small())")
    assert digests, "a call that is not its statement's plain value is digested as before"
