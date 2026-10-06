"""Around a statement, mutable module data is hashed only if the statement can change it.

The runtime takes the digest of every watched value before and after each
statement, so a change the statement makes is the notebook's own. A 128 MB
table, hashed in full around every ``x = 1``, cost 0.7 s a statement. The
environment and module constants are cheap and always taken; a mutable value
only when the statement reaches its module or names the value itself.
"""

import sys
import types

import pytest

from cash.notebook.recorded_reads import snapshot


@pytest.fixture
def statelib():
    module = types.ModuleType("statelib_snapshot_probe")
    module.__file__ = "/tmp/statelib_snapshot_probe.py"
    module.TABLE = list(range(1000))
    module.K = 3
    sys.modules[module.__name__] = module
    yield module
    del sys.modules[module.__name__]


def _watched(module):
    return [("mod", f"{module.__name__}.TABLE"), ("mod", f"{module.__name__}.K")]


def test_an_unrelated_statement_leaves_out_the_table_only(statelib):
    found = snapshot(_watched(statelib), "y = 1", {"y": 0, "statelib": statelib})
    assert set(found) == {("mod", f"{statelib.__name__}.K")}


@pytest.mark.parametrize(
    "code",
    [
        "statelib.TABLE[0] = 5",
        "arr[0] = 5",
        "get_ipython().run_line_magic('run', 'x.py')",
        "%run x.py",
    ],
    ids=["through-the-module", "through-a-name-bound-to-it", "a-magic", "not-python"],
)
def test_a_statement_that_can_change_the_table_takes_it(statelib, code):
    ns = {"statelib": statelib, "arr": statelib.TABLE}
    assert ("mod", f"{statelib.__name__}.TABLE") in snapshot(_watched(statelib), code, ns)


def test_without_code_everything_is_taken(statelib):
    assert len(snapshot(_watched(statelib))) == 2
