"""The closure check passes over numpy ``datetime64`` and ``timedelta64``
values as it does numpy's numbers: a list of 868,000 timedeltas
(``list(frame['t'].diff())``) was read through each one's reduce, two
calls an item, 27 s of a session. Pins the work, not the behaviour: a
refactor may expect this to fail."""

from __future__ import annotations

import numpy as np
import pytest

from cash import closure_sharing


@pytest.mark.parametrize("dtype", ["timedelta64[s]", "timedelta64[ns]", "datetime64[ms]"])
def test_no_time_is_reduced(monkeypatch, dtype):
    seen: list = []
    reduce = closure_sharing.reduce_counted

    def counted(item, protocol):
        seen.append(type(item))
        return reduce(item, protocol)

    monkeypatch.setattr(closure_sharing, "reduce_counted", counted)
    values = list(np.arange(5_000).astype(dtype))
    found = closure_sharing.closures_sharing({"deltas": values, "pair": (values[0], len)}, 5.0)
    assert found.names == [] and found.finished
    assert not {np.timedelta64, np.datetime64} & set(seen)
