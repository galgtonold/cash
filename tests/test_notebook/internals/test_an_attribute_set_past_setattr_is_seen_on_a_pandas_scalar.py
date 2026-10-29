"""``t.__dict__['tag'] = 1``, ``vars(t).update(...)`` and
``object.__setattr__(t, ...)`` on a pandas scalar turn on keeping them by
identity, as ``t.tag = 1`` does; the same writes on anything else do not.

The hook on the types' ``__setattr__`` never sees a write past it. The
statement's text is read for one instead: a name of the notebook is looked
at, and a write the text cannot pin down (a parameter of a function) turns
the switch on as it is. A class's own ``self.__dict__.update(...)`` does
not, unless the class is built on a pandas scalar. Nor does the hook see
a user's write of a name pandas keeps for itself (``td._seconds = 9``),
which any write of such a name, on anything, stands for.

Each case runs in a fresh interpreter: the switch is for the whole process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pandas")

CHECK = textwrap.dedent(
    """
    import ast, sys, types
    import pandas as pd
    from cash.notebook import shared_objects as so

    so.library_value_types()
    ns = {"t": pd.Timestamp("2021-01-01"), "o": types.SimpleNamespace(), "pd": pd}
    code = sys.argv[1]
    exec(code, ns)
    so.note_raw_attribute_writes(ast.parse(code), ns)
    print(so.scalar_identity())
    """
)

CASES = {
    "t.__dict__['tag'] = 1": True,
    "vars(t)['tag'] = 1": True,
    "vars(t).update(tag=1)": True,
    "object.__setattr__(t, 'tag', 1)": True,
    "def tag(x):\n    vars(x)['tag'] = 1\ntag(t)": True,
    "o.__dict__['tag'] = 1": False,
    "vars(o).update(tag=1)": False,
    "t.seconds if hasattr(t, 'seconds') else None\nvars(t)": False,
    "class C:\n    def __init__(self, **kw):\n        self.__dict__.update(kw)\nc = C(a=1)": False,
    "class T(pd.Timestamp):\n    def mark(self):\n        self.__dict__['m'] = 1": True,
    # A name pandas keeps for itself, which the hook lets through.
    "td = pd.Timedelta(5)\ntd._seconds = 9": True,
    "p = pd.Period('2020')\nsetattr(p, '_cache', {})": True,
    "td = pd.Timedelta(5)\nvars(td).update(_microseconds=1)": True,
    "td = pd.Timedelta(5)\ntd.__dict__['_seconds'] = 9": True,
    "td = pd.Timedelta(5)\ntd.seconds": False,
}


@pytest.mark.parametrize(("code", "on"), CASES.items(), ids=list(CASES))
def test_the_switch_turns_on_for_a_write_past_setattr_on_a_pandas_scalar(code, on):
    done = subprocess.run([sys.executable, "-c", CHECK, code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(on)
