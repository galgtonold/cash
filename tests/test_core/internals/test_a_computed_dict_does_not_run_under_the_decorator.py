"""A ``@cash.cache`` call whose global is an object with a computed
``__dict__`` (a property) never runs it, as plain Python never does.

Each walk of the key over the object read ``obj.__dict__``: the property ran
15 times on the call's first run and 7 more on a hit. The key is built from
the dict Python keeps for the object (``stored_dict``, in C), which its
attribute lookup reads, so a change to an attribute still moves the key.
"""

from __future__ import annotations

import sys
import textwrap
import warnings

import pytest

import cash
from cash import _native

pytestmark = pytest.mark.skipif(
    getattr(_native.kernels, "stored_dict", None) is None, reason=f"no C walk kernels: {_native.REASON}"
)

MODULE = textwrap.dedent(
    """
    RUNS = [0]

    class Lazy:
        def __init__(self):
            self.kept = 1

        @property
        def __dict__(self):
            RUNS[0] += 1
            return {"kept": 1}

    OBJ = Lazy()

    def f(x):
        return OBJ.kept + x
    """
)


@pytest.fixture
def lazy(tmp_path, monkeypatch):
    (tmp_path / "lazy_dict_global.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "lazy_dict_global", raising=False)
    import lazy_dict_global

    yield lazy_dict_global
    sys.modules.pop("lazy_dict_global", None)


def test_the_property_never_runs_and_a_change_is_seen(lazy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = cash.cache(lazy.f)
        assert f(1) == 2
        assert f(1) == 2
        assert lazy.RUNS[0] == 0
        lazy.OBJ.kept = 5
        assert f(1) == lazy.f(1) == 6
    assert lazy.RUNS[0] == 0
