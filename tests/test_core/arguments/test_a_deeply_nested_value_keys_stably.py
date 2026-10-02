"""A set however deep in an argument, and a container subclass's own state,
key the same in every process and as part of one walk.

The canonical form sorts every set it meets, so a set's iteration order, which
PYTHONHASHSEED picks, cannot reach the key. A part cut off at a fixed depth and
handed to pickle as it is keeps that order, and the key changes from process to
process: a persistent entry never hits again. A container subclass's own
attributes are walked with the container, so a list two of them share is
marked as shared and a loop back to the container is caught.

The walk keeps its path on the heap, so a value nested deeper than Python's
recursion limit is walked whole, on a small stack too: a recursive walk took
C stack for every level on Python 3.10 and overflowed Windows' 2 MB main
thread stack, killing the process. A value deeper than pickle follows, a
very long linked list say, runs the call uncached with a warning that says
so: it is never keyed on part of itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from cash import CashCacheIneffectiveWarning
from cash.canonical_form import stable_key_repr
from cash.content_hashers import BUILTIN_CONTENT

PROGRAM = textwrap.dedent("""
    import hashlib
    from cash.canonical_form import canonical_bytes
    from cash.content_hashers import BUILTIN_CONTENT

    class Tagged(dict):
        pass

    S = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}

    def wrapped(value, times):
        for _ in range(times):
            value = [value]
        return value

    in_state = Tagged(a=1)
    in_state.meta = wrapped(S, 6)
    values = {"in a subclass's state": in_state, "sixty lists down": wrapped(S, 60)}
    for name, value in values.items():
        print(name, "=", hashlib.sha256(canonical_bytes(value, BUILTIN_CONTENT)).hexdigest())
""")


def _keys(seed: str) -> dict[str, str]:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    done = subprocess.run([sys.executable, "-c", PROGRAM], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    return dict(line.split(" = ") for line in done.stdout.strip().splitlines())


@pytest.mark.timeout(300)
def test_a_deep_set_keys_alike_under_every_hash_seed():
    runs = [_keys(seed) for seed in ("1", "2", "3")]
    assert len(runs[0]) == 2
    assert runs[0] == runs[1] == runs[2]


class Tagged(dict):
    pass


def test_a_list_two_attributes_share_keys_apart_from_two_equal_lists(cash_instance):
    @cash_instance.cache
    def one_list(d):
        return d.a is d.b

    shared = Tagged()
    shared.a = shared.b = []
    separate = Tagged()
    separate.a, separate.b = [], []
    # Keyed alike, the second call is served the first one's True.
    assert one_list(shared) is True
    assert one_list(separate) is False


def test_a_subclass_whose_attribute_is_itself_runs_uncached(cash_instance):
    runs = []

    @cash_instance.cache
    def size(d):
        runs.append(1)
        return len(d)

    looped = Tagged(x=1)
    looped.me = looped
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert size(looped) == 1
        assert size(looped) == 1
    assert len(runs) == 2
    codes = [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert "KEY-UNHASHABLE-ARG" in codes


class Node:
    def __init__(self, nxt):
        self.nxt = nxt


def _chain(n):
    head = None
    for _ in range(n):
        head = Node(head)
    return head


def _nested(n):
    value = []
    for _ in range(n):
        value = [value]
    return value


#: Deeper than pickle follows on any supported Python: its recursion limit
#: up to 3.11, its C recursion limit on 3.12 and 3.13, the C stack on 3.14.
TOO_DEEP = 100_000


def _too_deep_warnings(caught) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, CashCacheIneffectiveWarning)
        and getattr(w.message, "code", None) == "KEY-UNHASHABLE-ARG"
    ]


@pytest.mark.parametrize(
    "make", [lambda: _chain(TOO_DEEP), lambda: _nested(TOO_DEEP)], ids=["linked list", "nested list"]
)
def test_a_value_deeper_than_pickle_follows_runs_uncached_as_too_deep(cash_instance, make):
    value = make()
    runs = []

    @cash_instance.cache
    def f(x):
        runs.append(1)
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(value) == 1
        assert f(value) == 1
    assert len(runs) == 2
    codes = [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert "KEY-BUILD-FAILED" not in codes
    messages = _too_deep_warnings(caught)
    assert messages
    assert all("nested too deeply to key" in m for m in messages)


def _keyed_once(cash_instance, value) -> bool:
    runs = []

    @cash_instance.cache
    def f(x):
        runs.append(1)
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f(value)
        f(value)
    return len(runs) == 1


def test_a_value_hundreds_of_levels_deep_is_cached(cash_instance):
    """Well past where a recursive walk stopped (about 230 levels under
    pytest), well inside what pickle follows on every Python."""
    assert _keyed_once(cash_instance, _nested(300))
    assert _keyed_once(cash_instance, _chain(200))
    with_a_set = _nested(300)
    with_a_set.append({"a", "b"})
    assert _keyed_once(cash_instance, with_a_set)


def test_a_deep_value_keys_apart_from_one_that_differs_at_the_bottom(cash_instance):
    runs = []

    @cash_instance.cache
    def f(x):
        runs.append(1)
        return len(runs)

    assert f(_nested(300)) == 1
    other = _nested(299)
    other.append(1)
    assert f(other) == 2


# Walks values far deeper than Python's recursion limit in a thread whose stack
# is a fraction of the main thread's on any platform (Windows gives python.exe
# 2 MB): the walk must hold its path on the heap. The values are built and freed
# on the main thread: freeing a long chain of objects recurses in C on some
# Pythons (3.13), whatever cash does.
STACK_PROGRAM = textwrap.dedent("""
    import sys, threading
    from cash.canonical_form import stable_key_repr
    from cash.content_hashers import BUILTIN_CONTENT

    class Node:
        def __init__(self, nxt, tags=None):
            self.nxt = nxt
            self.tags = tags

    def chain(n):
        head = None
        for i in range(n):
            head = Node(head, {"b", "a"} if i == 0 else None)
        return head

    def nested(n):
        value = [{"b", "a"}]
        for _ in range(n):
            value = [value]
        return value

    values = [chain(20000), nested(20000)]
    forms = []

    def walk_them():
        for value in values:
            forms.append(stable_key_repr(value, BUILTIN_CONTENT))

    threading.stack_size(256 * 1024)
    worker = threading.Thread(target=walk_them)
    worker.start()
    worker.join()
    print(len(forms))
""")


def test_the_walk_follows_a_value_of_any_depth_on_a_small_stack():
    done = subprocess.run([sys.executable, "-c", STACK_PROGRAM], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "2"


def _bottom(form):
    """The form at the bottom of a list nested in lists, and how deep it is.
    Unwrapped in a loop: comparing the whole form recurses."""
    depth = 0
    while isinstance(form, tuple) and form[:2] == ("__cash_type__", "list"):
        form = form[2][0]
        depth += 1
    return form, depth


def test_a_set_far_below_the_recursion_limit_is_still_sorted():
    """A set at the bottom of a value deeper than the recursion limit is
    walked to and sorted, not left to pickle in hash-seed order."""
    levels = sys.getrecursionlimit() * 3
    a = [{"x", "y", "z"}]
    b = [{"z", "y", "x"}]
    for _ in range(levels):
        a, b = [a], [b]
    bottom_a, depth_a = _bottom(stable_key_repr(a, BUILTIN_CONTENT))
    bottom_b, depth_b = _bottom(stable_key_repr(b, BUILTIN_CONTENT))
    assert depth_a == depth_b == levels + 1
    assert bottom_a == bottom_b == ("__cash_type__", "set", ("x", "y", "z"))
