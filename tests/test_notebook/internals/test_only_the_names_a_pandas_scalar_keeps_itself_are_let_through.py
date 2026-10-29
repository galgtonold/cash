"""Only the names a pandas scalar sets in its own dict pass the hook unseen.

Any ``_x`` was taken for the type's own cache when the type had an
attribute ``x``: a user's ``t._month = 'user'`` never turned the share
check's identity switch on, and a frame holding ``t`` was restored with a
copy that lacked it. The answers were also kept per class, which kept a
class made per call alive. Now a fixed set of names per type passes.

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
    import gc
    import sys
    import pandas as pd
    from cash.notebook import shared_objects as so

    so.library_value_types()
    td = pd.Timedelta(5)
    td.seconds, td.microseconds
    str(pd.Period("2020")), pd.Period("2020").start_time
    assert not so.scalar_identity_news(), "pandas' own cache turned the switch on"

    for i in range(5):
        type("K%d" % i, (pd.Timedelta,), {})(3).seconds
    gc.collect()
    assert not [c for c in pd.Timedelta.__subclasses__() if c.__name__.startswith("K")], "a class kept alive"

    name, made = sys.argv[1], {"ts": pd.Timestamp(1), "td": pd.Timedelta(1)}[sys.argv[2]]
    setattr(made, name, "user")
    assert so.scalar_identity_news(), name
    print("ok")
    """
)


@pytest.mark.parametrize(
    "name, kind", [("_month", "ts"), ("_date", "ts"), ("_day", "ts"), ("_days", "td"), ("tag", "td")]
)
def test_a_name_of_the_users_turns_the_switch_on(name, kind):
    done = subprocess.run([sys.executable, "-c", CHECK, name, kind], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
