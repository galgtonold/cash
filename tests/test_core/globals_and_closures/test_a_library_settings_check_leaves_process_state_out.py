"""What the library-settings check keeps is a setting, not state of the process.

Three ways a value that differs between processes, though no setting
changed, made every fresh process miss (`library_reads`):

- a counter numba bumps as it compiles, only to name what it makes
  (``ir_utils._unique_var_count``): an earlier call served from the cache
  skips its compile, so the next compiling call reads the counter lower;
- one dict reached under two names, read by one before a context manager
  set an entry and by the other inside it (xarray's ``infer_freq`` in
  ``option_context``): the value read inside was kept;
- a package the body imports lazily, filling a list of its submodule's as
  it loads (numba's cache prefixes): the submodule was done loading, so the
  list was kept as first read, empty, and the next process, whose lookup
  imports the package, reads it filled;
- a list threadpoolctl makes of a set of strings (``_ALL_PREFIXES``), in
  the order each process's string hashing gives it: every scikit-learn fit
  reads it.

Each still misses when the setting it reads is changed from outside.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]


def _run(script, *args, env=None):
    out = subprocess.run(
        [sys.executable, "-W", "ignore", str(script), *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
        env=env,
    )
    return out.stdout.split()


NUMBA = """
import sys
import numpy as np
import numba
from cash import Cash

c = Cash(cache_dir={cache!r}, register_magic=False)


@c.cache
def first(n):
    @numba.njit(parallel=True)
    def k(a):
        s = 0.0
        for i in numba.prange(a.shape[0]):
            s += a[i] * 2
        return s

    return k(np.arange(n, dtype=np.float64))


@c.cache
def second(n):
    @numba.njit(parallel=True)
    def k(a):
        s = 0.0
        for i in numba.prange(a.shape[0]):
            s += a[i] + 1
        return s

    return k(np.arange(n, dtype=np.float64)) * (3 if numba.config.BOUNDSCHECK else 1)


if sys.argv[1] == "set":
    numba.config.BOUNDSCHECK = 1
print(first(10), second(10), first.cache_info()["hits"], second.cache_info()["hits"])
"""


def test_a_counter_numba_names_ir_variables_by_is_no_setting(tmp_path):
    pytest.importorskip("numba")
    script = tmp_path / "jit.py"
    script.write_text(textwrap.dedent(NUMBA.format(cache=str(tmp_path / "cache"))), encoding="utf-8")
    assert _run(script, "keep") == ["90.0", "55.0", "0", "0"]
    # The first call is served, its compile skipped: the second compiles
    # with numba's variable counter lower than when it was stored. Served.
    assert _run(script, "keep") == ["90.0", "55.0", "1", "1"]
    # A numba setting changed: both run (numba's compile reads it too).
    assert _run(script, "set") == ["90.0", "165.0", "0", "0"]


CONFIG = """
_cfg = {"a": {"b": None, "c": 1}}


class context:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = _cfg["a"]["b"]
        _cfg["a"]["b"] = self.value

    def __exit__(self, *exc):
        _cfg["a"]["b"] = self.old
"""

PACKAGE = """
from .config import _cfg


def other():
    return _cfg["a"]["c"]


def using():
    return _cfg["a"]["b"]
"""

TWO_NAMES = """
import sys
sys.path.insert(0, {site!r})
import cash_ut_twonames as lib
import cash_ut_twonames.config as config
from cash import Cash


def body():
    lib.other()
    # Read under the config module's name before the context sets it, and
    # under the package's inside it.
    with config.context(False):
        inside = lib.using()
    return repr(inside)


if sys.argv[1] == "set":
    config._cfg["a"]["b"] = True
f = Cash(cache_dir={cache!r}, register_magic=False).cache(body)
print(f(), f.cache_info()["hits"])
"""


def test_a_dict_reached_under_two_names_is_kept_as_first_read_under_either(tmp_path):
    site = tmp_path / "venv" / "lib" / "site-packages"
    pkg = site / "cash_ut_twonames"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(PACKAGE, encoding="utf-8")
    (pkg / "config.py").write_text(CONFIG, encoding="utf-8")
    script = tmp_path / "two.py"
    script.write_text(
        textwrap.dedent(TWO_NAMES.format(site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8"
    )
    assert _run(script, "keep") == ["False", "0"]
    # A fresh process reads the entry as the first call did before the
    # context: served.
    assert _run(script, "keep") == ["False", "1"]
    # The entry set from outside before the call: it runs.
    assert _run(script, "set") == ["False", "0"]


LAZY_PACKAGE = """
from . import prefixes

prefixes.add("numba")


def value(x):
    return x + 1
"""

LAZY_PREFIXES = """
_index_prefixes = []


def add(prefix):
    if prefix not in _index_prefixes:
        _index_prefixes.append(prefix)
"""

LAZY = """
import sys
sys.path.insert(0, {site!r})
from cash import Cash


def body(x):
    import cash_ut_lazy

    return cash_ut_lazy.value(x)


f = Cash(cache_dir={cache!r}, register_magic=False).cache(body)
print(f(1), f.cache_info()["hits"])
"""


def test_a_list_a_package_fills_as_the_body_imports_it_is_read_once_loaded(tmp_path):
    site = tmp_path / "venv" / "lib" / "site-packages"
    pkg = site / "cash_ut_lazy"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(LAZY_PACKAGE, encoding="utf-8")
    (pkg / "prefixes.py").write_text(LAZY_PREFIXES, encoding="utf-8")
    script = tmp_path / "lazy.py"
    script.write_text(textwrap.dedent(LAZY.format(site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8")
    assert _run(script) == ["2", "0"]
    # Every later process is served, not every second one.
    assert _run(script) == ["2", "1"]
    assert _run(script) == ["2", "1"]


THREADPOOLS = """
import threadpoolctl
from cash import Cash


def body():
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        return 7


f = Cash(cache_dir={cache!r}, register_magic=False).cache(body)
print(f(), f.cache_info()["hits"])
"""


def test_a_list_made_from_a_set_in_each_process_s_order_is_no_setting(tmp_path):
    pytest.importorskip("threadpoolctl")
    import os

    script = tmp_path / "pools.py"
    script.write_text(textwrap.dedent(THREADPOOLS.format(cache=str(tmp_path / "cache"))), encoding="utf-8")
    seeds = [{**os.environ, "PYTHONHASHSEED": str(seed)} for seed in (1, 2, 3, 4)]
    assert _run(script, env=seeds[0]) == ["7", "0"]
    # Other string hashing, the controllers' names in another order: served.
    assert [_run(script, env=env) for env in seeds[1:]] == [["7", "1"]] * 3
