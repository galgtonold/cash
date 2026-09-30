"""A set held by a ``functools.partial``, a ``deque`` or an iterator keys the
same in every process.

Their contents live in C, not in a ``__dict__``, so the walk that sorts sets
never saw the set, and pickle wrote it in the order PYTHONHASHSEED picks: a
callback ``partial(score, allowed={...})`` passed to a cached function made a
new entry in every run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent("""
    import collections, functools, tempfile
    from cash import Cash, FileBackend

    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    def score(rows, allowed=()):
        return 0

    S = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
    values = {
        "partial arg": functools.partial(score, S),
        "partial keyword": functools.partial(score, allowed=S),
        "deque": collections.deque([S]),
        "iterator": iter([S, 1]),
    }
    for name, value in values.items():
        print(name, "=", c._args.hash_payload((value,), {}))
""")


def _keys(seed: str) -> dict[str, str]:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    done = subprocess.run([sys.executable, "-c", PROGRAM], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    return dict(line.split(" = ") for line in done.stdout.strip().splitlines())


@pytest.mark.timeout(300)
def test_the_key_does_not_depend_on_the_hash_seed():
    runs = [_keys(seed) for seed in ("1", "2", "3", "4")]
    assert len(runs[0]) == 4
    for name in runs[0]:
        assert len({run[name] for run in runs}) == 1, name


def test_the_set_s_content_still_counts():
    import collections
    import functools
    import tempfile

    from cash import Cash, FileBackend

    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    def key(v):
        return c._args.hash_payload((v,), {})

    assert key(functools.partial(print, {"a", "b"})) != key(functools.partial(print, {"a", "c"}))
    assert key(collections.deque([{"a"}])) != key(collections.deque([{"b"}]))
    assert key(functools.partial(print, {"a", "b"})) == key(functools.partial(print, {"b", "a"}))
