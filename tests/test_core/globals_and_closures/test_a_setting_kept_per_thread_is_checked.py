"""A library setting kept per thread is checked as the calling thread sees it.

scikit-learn keeps its configuration in a ``threading.local``
(``sklearn._config._threadlocal.global_config``), made on first use as a copy
of ``_global_config``; ``set_config`` and ``config_context`` change the
thread's copy in place and never ``_global_config``. The recorded reads
stopped at the per-thread holder, so a call was keyed only on
``_global_config``: after ``set_config(transform_output="pandas")`` a cached
``StandardScaler().fit_transform(X)`` served the ndarray made before it, and
a DataFrame stored inside ``config_context(...)`` was served after the block.

Now an attribute of a per-thread holder is read in the calling thread's
``__dict__``, and one the call itself made there (the thread had none yet)
is served while the thread still has none or holds what the call left.
Checked with real scikit-learn on the ordinary path, through the state memo
and on the fast hit path, in another thread and in new processes, and with a
library of the test's own keeping a setting in a ``threading.local``.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import threading
import time

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo
from tests._scripts import run_python

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")
preprocessing = pytest.importorskip("sklearn.preprocessing")

pytestmark = [
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

X = np.arange(6.0).reshape(3, 2)


def scale(X):
    return preprocessing.StandardScaler().fit_transform(X)


def output_setting():
    return sklearn.get_config()["transform_output"]


def kind(v):
    return type(v).__name__


@pytest.fixture(autouse=True)
def sklearn_config():
    """Whatever a test sets is put back as it leaves."""
    pytest.importorskip("pandas")
    with sklearn.config_context():
        yield


@pytest.fixture(params=["ordinary", "state memo", "fast"])
def path(request, monkeypatch):
    if request.param == "ordinary":
        monkeypatch.setattr(hit_memo, "ENABLED", False)
        monkeypatch.setattr(state_memo, "ENABLED", False)
    elif request.param == "state memo":
        if not state_memo.ENABLED:
            pytest.skip("the state memo needs the C extension")
        monkeypatch.setattr(hit_memo, "ENABLED", False)
    elif not hit_memo.ENABLED:
        pytest.skip("the fast hit path needs the C extension")
    return request.param


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _served(cached, *args):
    """Call until a call is served without running the body; return its
    result."""
    for _ in range(3):
        cached(*args)
    hits = cached.cache_info()["hits"]
    out = cached(*args)
    assert cached.cache_info()["hits"] == hits + 1, cached.cache_info()["miss_reasons"]
    return out


def _served_fast(cached, *args):
    """Call until served from the fast path (a positive control: on that
    path, a check that never ran there proves nothing)."""
    deadline = time.monotonic() + 20 * hit_memo.HitMemo.RESCAN_AFTER_S + 1.0
    while True:
        before = hit_memo.status()["fast_hits"]
        out = cached(*args)
        if hit_memo.status()["fast_hits"] > before:
            return out
        assert time.monotonic() < deadline, ("not served from the fast path", hit_memo.status())


def test_set_config_between_two_calls_misses_and_returns_what_plain_python_does(c, path):
    cached = c.cache(scale)
    assert kind(_served(cached, X)) == "ndarray"
    hits = cached.cache_info()["hits"]
    sklearn.set_config(transform_output="pandas")
    assert kind(cached(X)) == kind(scale(X)) == "DataFrame"
    assert cached.cache_info()["hits"] == hits, "served the ndarray made under the default output"
    assert "library setting changed" in cached.cache_info()["miss_reasons"]
    # The entry made under the pandas output is served while it holds.
    assert kind(_served(cached, X)) == "DataFrame"
    sklearn.set_config(transform_output="default")
    hits = cached.cache_info()["hits"]
    assert kind(cached(X)) == "ndarray"
    assert cached.cache_info()["hits"] == hits


def test_get_config_read_in_the_function_is_checked_on_every_path(c, path):
    cached = c.cache(output_setting)
    assert _served(cached) == "default"
    if path == "fast":
        assert _served_fast(cached) == "default"
    sklearn.set_config(transform_output="pandas")
    assert cached() == "pandas"
    assert "library setting changed" in cached.cache_info()["miss_reasons"]
    if path == "fast":
        assert _served_fast(cached) == "pandas"
    sklearn.set_config(transform_output="default")
    assert cached() == "default"


def test_a_result_stored_inside_config_context_is_not_served_after_the_block(c, path):
    cached = c.cache(scale)
    with sklearn.config_context(transform_output="pandas"):
        assert kind(_served(cached, X)) == "DataFrame"
    hits = cached.cache_info()["hits"]
    assert kind(cached(X)) == kind(scale(X)) == "ndarray"
    assert cached.cache_info()["hits"] == hits, "served the DataFrame stored inside the block"
    with sklearn.config_context(transform_output="pandas"):
        assert kind(cached(X)) == "DataFrame"


def _in_new_thread(fn, *args):
    out = []
    t = threading.Thread(target=lambda: out.append(fn(*args)))
    t.start()
    t.join()
    return out[0]


def test_each_thread_is_checked_on_its_own_config(c):
    cached = c.cache(output_setting)
    sklearn.set_config(transform_output="pandas")
    assert _served(cached) == "pandas"
    # A new thread starts from the defaults, as plain Python there does.
    assert _in_new_thread(cached) == _in_new_thread(output_setting) == "default"


def test_a_config_the_call_made_on_a_new_thread_is_served_until_it_changes(c):
    # The first call on a thread makes the thread's copy of the config: the
    # entry is served on a thread with none yet, or with the same one.
    cached = c.cache(output_setting)

    def twice():
        return cached(), cached()

    assert _in_new_thread(twice) == ("default", "default")
    hits = cached.cache_info()["hits"]
    assert _in_new_thread(cached) == "default"
    assert cached.cache_info()["hits"] == hits + 1
    sklearn.set_config(transform_output="default")
    assert cached() == "default"
    assert cached.cache_info()["hits"] == hits + 2

    def changed():
        sklearn.set_config(transform_output="pandas")
        return cached()

    assert _in_new_thread(changed) == "pandas"
    assert cached.cache_info()["hits"] == hits + 2


def test_a_call_that_sets_the_config_itself_is_not_served_the_value_it_read_before(c):
    # The first call on a thread makes the thread's config and then changes
    # it: the entry was made under the defaults, and the thread now holds
    # the pandas output, as plain Python's next call reads it.
    def read_then_set():
        out = sklearn.get_config()["transform_output"]
        sklearn.set_config(transform_output="pandas")
        return out

    cached = c.cache(read_then_set)

    def twice():
        # A new thread: its config dies with it.
        return cached(), cached()

    assert _in_new_thread(twice) == ("default", "pandas")


SCRIPT = """
import sys
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from cash import Cash

c = Cash(cache_dir={cache!r}, register_magic=False)


def scale(X):
    return StandardScaler().fit_transform(X)


if sys.argv[1] != "unset":
    sklearn.set_config(transform_output=sys.argv[1])
f = c.cache(scale)
print(type(f(np.arange(6.0).reshape(3, 2))).__name__, f.cache_info()["hits"])
"""


def test_the_config_is_checked_in_new_processes(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(SCRIPT.format(cache=str(tmp_path / "cache"))), encoding="utf-8")

    def run(setting):
        return run_python(script, setting, cwd=tmp_path, cache_dir=tmp_path / "cache").stdout.split()

    assert run("unset") == ["ndarray", "0"]
    assert run("unset") == ["ndarray", "1"]
    # The same config set before the call: the same result, served.
    assert run("default") == ["ndarray", "1"]
    assert run("pandas") == ["DataFrame", "0"]
    assert run("pandas") == ["DataFrame", "1"]


LIBRARY = """
import threading

_local = threading.local()
DEFAULTS = {"scale": 2}


def _config():
    if not hasattr(_local, "config"):
        _local.config = dict(DEFAULTS)
    return _local.config


def set_scale(v):
    _config()["scale"] = v


def scaled(x):
    return x * _config()["scale"]


def level():
    return _local.level if hasattr(_local, "level") else 0


def set_level(v):
    _local.level = v


def leveled(x):
    return x + level()
"""


@pytest.fixture
def lib(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_tlslib.py").write_text(LIBRARY, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    sys.modules.pop("cash_ut_tlslib", None)
    importlib.invalidate_caches()
    yield importlib.import_module("cash_ut_tlslib")
    sys.modules.pop("cash_ut_tlslib", None)


def test_a_library_setting_held_in_a_threading_local_is_checked(c, path, lib):
    cached = c.cache(lib.scaled)
    assert _served(cached, 3) == 6
    lib.set_scale(5)
    assert cached(3) == lib.scaled(3) == 15
    assert "library setting changed" in cached.cache_info()["miss_reasons"]
    assert _served(cached, 3) == 15


def test_an_attribute_a_threading_local_gets_later_is_checked(c, path, lib):
    cached = c.cache(lib.leveled)
    assert _served(cached, 1) == 1
    lib.set_level(4)
    assert cached(1) == lib.leveled(1) == 5
    assert _in_new_thread(cached, 1) == 1
