"""A once-only warning is once per code, so one warning never silences another.

The dedup key used to be ``(category, function, arg type)``. Four codes share
``(CashCacheIneffectiveWarning, f, "")`` -- CACHE-ASYNC-GENERATOR,
CACHE-IDENTITY-COUPLED, KEY-DEPENDS-ON-OPAQUE and CACHE-IF-BYPASSED -- so
whichever fired first hid the rest for the life of the process.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash, CashCacheIneffectiveWarning

pytestmark = pytest.mark.core

CODES = ("CACHE-IDENTITY-COUPLED", "KEY-DEPENDS-ON-OPAQUE", "CACHE-IF-BYPASSED")


def test_each_code_is_shown_once(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache
    def f(x):
        return x

    name = c.get_func_key(f)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(2):  # the second round is deduplicated
            for code in CODES:
                c._warn_once(CashCacheIneffectiveWarning, name, "", "something happened.", code=code, fix="fix it.")

    assert [w.message.code for w in rec] == list(CODES)
    assert [e["code"] for e in f.cache_info()["warnings"]] == list(CODES)
