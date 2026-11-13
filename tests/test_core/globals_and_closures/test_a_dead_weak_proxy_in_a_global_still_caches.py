"""A global holding a weak proxy whose object is gone still caches.

numba's dispatchers keep a ``weakref.proxy`` in an attribute after its
object is gone; librosa's ``yin``, ``cqt``, ``beat_track`` and
``chroma_stft`` reach one through a helper's global, and every read of it
raises ``ReferenceError`` -- ``isinstance`` too. The key build gave up
(KEY-BUILD-FAILED) and they ran uncached in every run. A dead proxy holds
nothing: the search for user code skips it, and the value channel keys it
as a dead proxy, the same in every process.
"""

from __future__ import annotations

import textwrap
import weakref

from cash.value_types import is_dead_proxy
from tests._scripts import run_python

JOB = textwrap.dedent("""
    import argparse
    import threading
    import types
    import warnings
    import weakref

    import cash

    class Thing:
        pass

    t = Thing()
    # A record of plain values: keyed by its content.
    CONF = types.SimpleNamespace(p=weakref.proxy(t), scale=3)
    # A library object cash cannot hash (it holds a lock): searched for
    # user code, with a warning.
    HOLDER = argparse.Namespace(lock=threading.Lock(), inner=[types.SimpleNamespace(p=weakref.proxy(t))], scale=5)
    del t

    def helper(x):
        return x * CONF.scale

    @cash.cache
    def through_record(x):
        return helper(x)

    @cash.cache
    def through_library_object(x):
        return x * HOLDER.scale

    fns = [through_record, through_library_object]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        print(" ".join(str(f(2)) for f in fns))
    print(" ".join(str(f.cache_info()["hits"]) for f in fns))
    print(sorted({str(w.message).split("]")[0].lstrip("[") for w in caught if "KEY-" in str(w.message)}))
""")


def test_a_dead_weak_proxy_does_not_stop_the_key(tmp_path):
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    first = run_python("job.py", cwd=tmp_path).stdout.splitlines()
    assert first[0].split() == ["6", "10"]
    assert "KEY-BUILD-FAILED" not in first[2]
    second = run_python("job.py", cwd=tmp_path).stdout.splitlines()
    assert second[0].split() == ["6", "10"]
    assert second[1].split() == ["1", "1"]


def test_is_dead_proxy_tells_a_dead_proxy_from_a_live_one():
    class Thing:
        pass

    t = Thing()
    proxy = weakref.proxy(t)
    assert not is_dead_proxy(proxy)
    assert not is_dead_proxy(t)
    del t
    assert is_dead_proxy(proxy)
