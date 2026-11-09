"""A cached function that sets a registry the process keeps runs every
time, and says so.

``np.seterr(divide='ignore')``, ``logging.getLogger("lib").setLevel(10)``,
``sys.setrecursionlimit(n)``: no variable holds what they change, so a hit
did not change it again. In a new process (here, the registry put back as
one finds it) the call was served and the setting never made, without a
word (`cash.tracking.process_registries`).
"""

from __future__ import annotations

import logging
import sys
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpurityWarning


def _np_seterr(x):
    import numpy as np

    old = np.seterr(divide="ignore")
    return x, old["divide"]


def _np_divide():
    import numpy as np

    return np.geterr()["divide"]


def _np_reset():
    import numpy as np

    np.seterr(divide="warn")


def _set_level(x):
    logging.getLogger("zz.cached").setLevel(10)
    return x + 1


def _set_limit(x):
    sys.setrecursionlimit(4321)
    return x + 1


CASES = {
    "np-seterr": (_np_seterr, _np_divide, _np_reset, "ignore"),
    "logger-level": (
        _set_level,
        lambda: logging.getLogger("zz.cached").level,
        lambda: logging.getLogger("zz.cached").setLevel(0),
        10,
    ),
    "recursionlimit": (_set_limit, sys.getrecursionlimit, lambda: sys.setrecursionlimit(1000), 4321),
}


@pytest.mark.parametrize("kind", list(CASES))
def test_the_setting_is_made_on_every_call(tmp_path, kind):
    if kind == "np-seterr":
        pytest.importorskip("numpy")
    func, read, reset, made = CASES[kind]
    limit = sys.getrecursionlimit()
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    cached = c.cache(func)
    try:
        reset()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            first = cached(1)
            for _ in range(2):
                # As a new process finds it.
                reset()
                assert cached(1) == first
                assert read() == made, "served without making the setting"
    finally:
        reset()
        sys.setrecursionlimit(limit)
    said = [str(w.message) for w in record if issubclass(type(w.message), CashImpurityWarning)]
    assert any("no variable holds" in text for text in said), said
