"""``td.seconds`` on 200,000 fresh Timedeltas does not pay for cash's hook.

An attribute set on a pandas scalar turns on keeping them by identity, so
cash hooks their ``__setattr__``. Every set went through the hook, the
type's own too: ``td.seconds`` keeps ``_seconds`` and ``_microseconds`` on
first read, and each asked again whether the name was one the type keeps
(4x plain). The answer is remembered per type and name, and once the
switch is on the hooks come off, as nothing is left for them to tell.

Run in a fresh interpreter: the switch is for the whole process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pandas")

CHECK = textwrap.dedent(
    """
    import pandas as pd
    from cash.notebook import shared_objects as so

    so.library_value_types()
    assert "__setattr__" in vars(pd.Timedelta)

    class Mine(pd.Timedelta):
        pass

    for _ in range(3):
        t = pd.Timedelta(seconds=7)
        assert t.seconds == 7
        assert Mine(seconds=1).seconds == 1
    assert not so.scalar_identity(), "the type's own cache is no attribute of the user's"

    t.tag = "x"
    assert so.scalar_identity()
    for kind in (pd.Timestamp, pd.Timedelta, pd.Period, type(pd.NaT), type(pd.NA)):
        assert "__setattr__" not in vars(kind) and "__delattr__" not in vars(kind), kind
    u = pd.Timedelta(seconds=2)
    u.tag = "y"
    assert u.tag == "y" and u.seconds == 2
    del u.tag
    print("ok")
    """
)


def test_the_hook_reads_the_types_own_names_once_and_comes_off():
    done = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
