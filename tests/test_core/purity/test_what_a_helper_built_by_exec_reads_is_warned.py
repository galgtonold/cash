"""What a helper built by ``exec`` into a module reads is warned about.

``exec("def stamp(x):\\n    return x + time.time()", globals())``: the
function has no source, so the analysis that finds a clock read, a fresh id
or ``os.environ["MODE"]`` in a helper's text found nothing, and the call was
stored with no word: the next run, in this process or a new one, was served
the first time, id or mode. Its bytecode shows when it may read something
the key misses (an attribute, an import, a call of anything but its
arguments and plain builtins), and then the call warns KEY-AMBIENT-READ, as
for a helper whose source names the read. A helper built so that only
computes on its arguments, or calls a function the walk reads, is cached
with no warning, and an edit to its text is still seen.
"""

from __future__ import annotations

import os  # noqa: F401 - read by the helpers the exec below builds
import time  # noqa: F401
import uuid  # noqa: F401
import warnings

import pytest

from cash import Cash

exec("def stamp(x):\n    return x + time.time()\n")
exec("def token(x):\n    return str(x) + uuid.uuid4().hex\n")
exec("def mode(x):\n    return x + os.environ['CASH_T_EXEC_MODE']\n")
exec("def pure(x):\n    return x * 2 + len(str(x))\n")
exec("def via_sourced(x):\n    return sourced(x) + 1\n")


def sourced(x):
    return x * 3


def calls_stamp(x):
    return stamp(x)  # noqa: F821 - defined by the exec above


def calls_token(x):
    return token(x)  # noqa: F821


def calls_mode(x):
    return mode(x)  # noqa: F821


def calls_pure(x):
    return pure(x)  # noqa: F821


def calls_via_sourced(x):
    return via_sourced(x)  # noqa: F821


def _calls(tmp_path, fn, arg):
    cached = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(fn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(arg)
        cached(arg)
    said = [str(w.message) for w in caught if "KEY-AMBIENT-READ" in str(w.message)]
    return cached, said


@pytest.mark.parametrize(
    ("fn", "arg"), [(calls_stamp, 1.0), (calls_token, "a"), (calls_mode, "a")], ids=["clock", "uuid", "environ"]
)
def test_a_read_it_may_make_is_warned(tmp_path, monkeypatch, fn, arg):
    monkeypatch.setenv("CASH_T_EXEC_MODE", "x")
    _, said = _calls(tmp_path, fn, arg)
    assert said, "the first value was stored with no word"
    assert "built by exec()" in said[0]


@pytest.mark.parametrize("fn", [calls_pure, calls_via_sourced], ids=["arguments only", "calls a read function"])
def test_one_that_reads_nothing_unseen_is_cached_quietly(tmp_path, fn):
    cached, said = _calls(tmp_path, fn, 3)
    assert said == []
    assert cached.cache_info()["hits"] == 1


def test_assume_safe_accepts_the_read(tmp_path):
    cached = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(calls_stamp, assume_safe=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = cached(1.0)
        assert cached(1.0) == first
    assert not [w for w in caught if "KEY-AMBIENT-READ" in str(w.message)]


def test_an_edit_to_its_text_is_seen(tmp_path):
    ns = {"__name__": __name__}
    exec("def edited(x):\n    return x + 1\n", ns)
    globals()["edited"] = ns["edited"]
    try:
        c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

        @c.cache
        def calls_edited(x):
            return edited(x)  # noqa: F821

        assert calls_edited(1) == 2
        exec("def edited(x):\n    return x + 2\n", ns)
        globals()["edited"] = ns["edited"]
        assert calls_edited(1) == 3
    finally:
        globals().pop("edited", None)
