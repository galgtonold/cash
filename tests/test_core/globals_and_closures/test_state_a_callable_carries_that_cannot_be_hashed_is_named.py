"""A callable whose state cannot be hashed is still cached, and never silently.

``GET_OBJECT = s3.get_object``, ``PLOT = ax.plot``, ``SUBMIT = pool.submit``:
a library object's bound method kept under a module-level name. The object
it is bound to holds locks, sockets or threads, so nothing of it can go into
the key. The call was cached by the method's name alone and nothing said so;
nor did anything for a module-level partial of the user's function over a
generator, called by name. They are still cached, and KEY-UNHASHABLE-GLOBAL
now names what was left out, once per function and name -- as it already did
for the user's own client object, a class-based decorator holding a lock, an
object holding a generator, a ``joblib.Parallel`` and a partial of a library
function over a generator.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import warnings

import pytest

pytestmark = pytest.mark.core

CODE = "KEY-UNHASHABLE-GLOBAL"

MODULE = textwrap.dedent("""
    import functools
    import threading
    from concurrent.futures import ThreadPoolExecutor

    RUNS = []

    # -- a library object's bound method under a module-level name ---------
    import boto3
    S3 = boto3.client("s3", region_name="eu-central-1", aws_access_key_id="k", aws_secret_access_key="s")
    GET_OBJECT = S3.get_object

    def s3_alias(x):
        RUNS.append(x)
        if x < 0:
            return GET_OBJECT(Bucket="b", Key="k")
        return x + 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG, AX = plt.subplots()
    PLOT = AX.plot

    def plot_alias(x):
        RUNS.append(x)
        return len(PLOT([x, x + 1]))

    POOL = ThreadPoolExecutor(max_workers=1)
    SUBMIT = POOL.submit

    def pool_alias(x):
        RUNS.append(x)
        return SUBMIT(abs, x).result()

    # -- the patterns that warned all along --------------------------------
    import requests

    class Client:
        def __init__(self, base):
            self.base = base
            self.session = requests.Session()
            self.lock = threading.Lock()
        def url(self, x):
            return f"{self.base}/{x}"

    API = Client("https://example.invalid")

    def own_client(x):
        RUNS.append(x)
        return API.url(x)

    class RateLimited:
        def __init__(self, fn):
            self.fn = fn
            self.lock = threading.Lock()
        def __call__(self, x):
            with self.lock:
                return self.fn(x)

    @RateLimited
    def limited(x):
        return x * 2

    def class_decorator(x):
        RUNS.append(x)
        return limited(x)

    class Feed:
        def __init__(self):
            self.rows = (i for i in range(3))
        def scale(self, x):
            return x * 3

    FEED = Feed()

    def object_with_a_generator(x):
        RUNS.append(x)
        return FEED.scale(x)

    import joblib
    PAR = joblib.Parallel(n_jobs=1)

    def joblib_parallel(x):
        RUNS.append(x)
        return PAR(joblib.delayed(abs)(v) for v in (x, -x))

    def add(x, rows):
        return x + 4

    ADD = functools.partial(add, rows=(i for i in range(3)))

    def partial_with_a_generator(x):
        RUNS.append(x)
        return ADD(x)

    LARGEST = functools.partial(max, default=(i for i in range(3)))

    def library_partial(x):
        RUNS.append(x)
        return LARGEST([x, x + 1])
""")

SILENT_BEFORE = {
    "s3_alias": ("GET_OBJECT", "S3"),
    "plot_alias": ("PLOT", "Axes"),
    "pool_alias": ("SUBMIT", "ThreadPoolExecutor"),
    "partial_with_a_generator": ("ADD", "functools.partial"),
}
WARNED_ALL_ALONG = {
    "own_client": "API",
    "class_decorator": "limited",
    "object_with_a_generator": "FEED",
    "joblib_parallel": "PAR",
    "library_partial": "LARGEST",
}


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"carriers_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module(name)
    yield module
    module.POOL.shutdown()
    sys.modules.pop(name, None)


def _called_three_times(cash_instance, mod, func):
    """(the KEY-UNHASHABLE-GLOBAL messages, how often the body ran)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = cash_instance.cache(getattr(mod, func))
        mod.RUNS.clear()
        first = fn(2)
        assert fn(2) == first
        assert fn(2) == first
    said = [str(w.message) for w in caught if CODE in str(w.message)]
    return said, len(mod.RUNS)


@pytest.mark.parametrize("func", sorted(SILENT_BEFORE))
def test_what_was_silent_is_cached_and_named(cash_instance, mod, func):
    name, owner = SILENT_BEFORE[func]
    said, runs = _called_three_times(cash_instance, mod, func)
    assert runs == 1, "still cached: the body ran once in three calls"
    assert len(said) == 1, said
    assert f"{name}'" in said[0]
    assert owner in said[0], "says what was left out of the key"
    assert "left out of the key" in said[0]


@pytest.mark.parametrize("func", sorted(WARNED_ALL_ALONG))
def test_what_warned_all_along_warns_once_and_is_cached(cash_instance, mod, func):
    said, runs = _called_three_times(cash_instance, mod, func)
    assert runs == 1
    assert len(said) == 1, said
    assert WARNED_ALL_ALONG[func] in said[0]


def test_each_function_reading_the_name_is_told(cash_instance, mod):
    source = "def second_reader(x):\n    RUNS.append(x)\n    return SUBMIT(abs, x).result() + 1\n"
    exec(compile(source, mod.__file__, "exec"), vars(mod))
    first, _ = _called_three_times(cash_instance, mod, "pool_alias")
    second, runs = _called_three_times(cash_instance, mod, "second_reader")
    assert len(first) == 1 and len(second) == 1
    assert "second_reader" in second[0]
    assert runs == 1


QUIET = textwrap.dedent("""
    import sys
    import threading

    RATES = {'a': 1}
    LOOKUP = RATES.get
    LOCK = threading.Lock()
    ACQUIRE = LOCK.acquire
    RELEASE = LOCK.release
    DONE = threading.Event()
    IS_SET = DONE.is_set
    FLUSH = sys.stdout.flush

    def rate(k):
        return LOOKUP(k)

    def under_a_lock(k):
        ACQUIRE()
        try:
            return RATES[k]
        finally:
            RELEASE()

    def asks_an_event(k):
        return RATES[k], IS_SET()

    def flushes(k):
        FLUSH()
        return RATES[k]
""")


@pytest.mark.parametrize("func", ["rate", "under_a_lock", "asks_an_event", "flushes"])
def test_a_method_whose_object_is_no_input_gives_no_warning(cash_instance, tmp_path, monkeypatch, func):
    """A dict's ``get`` hashes; a lock or an event is keyed by its kind, as
    when read as a global; a stream's ``flush`` holds output, not input."""
    name = f"quiet_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(QUIET, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module(name)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn = cash_instance.cache(getattr(module, func))
            before = fn("a")
            module.RATES["a"] = 2
            assert fn("a") != before, "the dict it reads is still keyed"
        assert not [w for w in caught if CODE in str(w.message)]
    finally:
        sys.modules.pop(name, None)
