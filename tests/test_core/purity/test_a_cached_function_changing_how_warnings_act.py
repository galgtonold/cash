"""A cached function that leaves the warnings module set otherwise runs
every time, and says so; one that puts it back as it found it is stored.

``warnings.formatwarning = f`` (SALib's ``delta.analyze``) changes how
every later warning of the process reads, ``warnings.simplefilter("once")``
and ``warnings.defaultaction`` whether it shows at all: a hit would not
make the change, and the program's later warnings would differ from plain
Python's (`cash.tracking.process_registries`). ``with
warnings.catch_warnings(): warnings.simplefilter("ignore")`` leaves
nothing behind, so its calls are stored and served.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpurityWarning


def _custom_format(message, category, filename, lineno, line=None):
    return f"CUSTOM: {message}\n"


#: What `_custom_show` was handed: once it is set, cash's own warnings go there.
SHOWN: list = []


def _custom_show(message, category, filename, lineno, file=None, line=None):
    SHOWN.append(message)


def _set_format(x):
    warnings.formatwarning = _custom_format
    return x + 1


def _set_show(x):
    warnings.showwarning = _custom_show
    return x + 1


def _set_default_action(x):
    warnings.defaultaction = "ignore"
    return x + 1


def _later_warning_text():
    return warnings.formatwarning("later", UserWarning, "here.py", 1)


CASES = {
    "formatwarning": (_set_format, "formatwarning", _custom_format),
    "showwarning": (_set_show, "showwarning", _custom_show),
    "defaultaction": (_set_default_action, "defaultaction", "ignore"),
}


@pytest.mark.parametrize("kind", list(CASES))
def test_the_change_is_made_on_every_call(tmp_path, kind):
    func, name, made = CASES[kind]
    found = getattr(warnings, name)
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    cached = c.cache(func)
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            first = cached(1)
            for _ in range(2):
                # As a new process finds it.
                setattr(warnings, name, found)
                assert cached(1) == first
                assert getattr(warnings, name) is made, "served without making the change"
            if kind == "formatwarning":
                assert _later_warning_text() == "CUSTOM: later\n"
    finally:
        setattr(warnings, name, found)
    messages = [w.message for w in record] + SHOWN
    said = [str(m) for m in messages if isinstance(m, CashImpurityWarning)]
    assert any("no variable holds" in text for text in said), said
    assert cached.cache_info()["hits"] == 0


def _quiet(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("not shown", UserWarning, stacklevel=1)
    return x * 2


def _left_once(x):
    warnings.simplefilter("once", UserWarning)
    return x - 1


def test_filters_put_back_on_exit_are_stored(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    cached = c.cache(_quiet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CashImpurityWarning)
        before = list(warnings.filters)
        assert cached(3) == 6
        assert cached(3) == 6
        assert warnings.filters == before
    info = cached.cache_info()
    assert (info["hits"], info["misses"]) == (1, 1), info


def test_filters_left_changed_are_made_on_every_call(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    cached = c.cache(_left_once)
    entry = ("once", None, UserWarning, None, 0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        first = cached(3)
        for _ in range(2):
            # As a new process finds them.
            while entry in warnings.filters:
                warnings.filters.remove(entry)
            assert cached(3) == first
            assert warnings.filters[0] == entry, "served without changing the filters"
    said = [str(w.message) for w in record if issubclass(type(w.message), CashImpurityWarning)]
    assert any("the warnings filters" in text for text in said), said
    assert cached.cache_info()["hits"] == 0
