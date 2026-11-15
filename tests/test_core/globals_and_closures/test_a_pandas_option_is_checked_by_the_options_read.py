"""A stored call is checked on the pandas options it read, not on all of them.

pandas keeps its options in one nested dict (``_global_config``, bound under
several names). Checking the whole dict on lookup missed whenever any option
differed: a notebook kernel's defaults differ from a script's
(``display.max_columns`` 20 against 0, ``display.encoding``), and a module
that registers options of its own adds entries. A script's entries were never
served to a notebook, nor after such an import. Now a call is checked on the
entries its code read: by constant keys (``config["display"]["precision"]``)
or by the key handed to the option getter (``pd.get_option``,
``pd.options.display.precision``).

Each scenario checks the cached call returns what plain Python does, on the
ordinary path, through the state memo and on the fast hit path, and in new
processes.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import time

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

pd = pytest.importorskip("pandas")

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.skipif(
        sys.version_info < (3, 12), reason="a body's reads are recorded with sys.monitoring, Python 3.12 or newer"
    ),
]

#: A library that registers an option of its own when imported, as pandas
#: plugins do.
PLUGIN = """
import importlib

_cf = importlib.import_module("pandas._config.config")
_cf.register_option("cash_ut_plugin_{n}.level", 1)
"""


def by_get_option(x):
    return round(x / 7, pd.get_option("display.precision"))


def by_options(x):
    return round(x / 7, pd.options.display.precision)


def by_frame(x):
    # pandas' own code reads its options by constant keys.
    return pd.DataFrame({"a": [x / 7, 1.0]}).to_string()


#: Served from the fast path. pandas 2's ``get_option`` and ``pd.options``
#: (an object's attribute) are not: each is still checked to return what
#: plain Python does there.
FAST = {by_frame} | ({by_get_option} if int(pd.__version__.split(".")[0]) >= 3 else set())


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


@pytest.fixture
def options():
    with pd.option_context("display.precision", 6, "display.max_columns", 0):
        yield


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    count = [0]

    def load():
        count[0] += 1
        name = f"cash_ut_pd_plugin_{tmp_path.name.replace('-', '_')}_{count[0]}"
        (site / f"{name}.py").write_text(PLUGIN.format(n=f"{tmp_path.name}_{count[0]}".replace("-", "_")))
        importlib.invalidate_caches()
        return importlib.import_module(name)

    monkeypatch.syspath_prepend(str(site))
    return load


def _settle(cached, plain, path, *args):
    """Call until served without running the body; on the fast path, until
    served from the memo, for one it serves (`FAST`)."""
    for _ in range(3):
        assert cached(*args) == plain(*args)
    if path != "fast" or plain not in FAST:
        return
    deadline = time.monotonic() + 5 * hit_memo.HitMemo.RESCAN_AFTER_S + 1.0
    while True:
        before = hit_memo.status()["fast_hits"]
        assert cached(*args) == plain(*args)
        if hit_memo.status()["fast_hits"] > before:
            return
        assert time.monotonic() < deadline, "not served from the fast path: the test proves nothing"
        time.sleep(0.01)


@pytest.mark.parametrize("plain", [by_get_option, by_options, by_frame], ids=lambda f: f.__name__)
def test_an_option_read_is_checked(c, path, options, plain):
    cached = c.cache(plain)
    _settle(cached, plain, path, 1)
    hits = cached.cache_info()["hits"]
    pd.set_option("display.precision", 2)
    assert cached(1) == plain(1)
    assert cached.cache_info()["hits"] == hits, "served the result made under the other precision"
    assert "library setting changed" in str(cached.cache_info())
    # The entry stored now is the one made under precision 2.
    pd.set_option("display.precision", 6)
    assert cached(1) == plain(1)
    assert cached.cache_info()["hits"] == hits
    assert cached(1) == plain(1)
    assert cached.cache_info()["hits"] == hits + 1


@pytest.mark.parametrize("plain", [by_get_option, by_options, by_frame], ids=lambda f: f.__name__)
def test_an_option_not_read_or_a_registered_one_changes_nothing(c, path, options, plugin, plain):
    cached = c.cache(plain)
    _settle(cached, plain, path, 1)
    hits = cached.cache_info()["hits"]
    pd.set_option("display.html.border", 3)
    try:
        plugin()
        assert cached(1) == plain(1)
    finally:
        pd.reset_option("display.html.border")
    assert cached.cache_info()["hits"] == hits + 1, cached.cache_info()["miss_reasons"]


SCRIPT = """
import sys
sys.path.insert(0, {site!r})
import pandas as pd
from cash import Cash

c = Cash(cache_dir={cache!r}, register_magic=False)


def by_options(x):
    return round(x / 7, pd.options.display.precision)


def by_frame(x):
    return pd.DataFrame({{"a": [x / 7, 1.0]}}).to_string()


precision, columns, extra = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
pd.set_option("display.precision", precision)
pd.set_option("display.max_columns", columns)
if extra:
    import cash_ut_pd_plugin
f, g = c.cache(by_options), c.cache(by_frame)
print(repr(f(1) == by_options(1) and g(1) == by_frame(1)), f.cache_info()["hits"] + g.cache_info()["hits"])
"""


def test_options_are_checked_in_a_new_process(tmp_path):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_pd_plugin.py").write_text(PLUGIN.format(n="restart"), encoding="utf-8")
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(SCRIPT.format(site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8")

    def run(precision, columns, extra):
        out = subprocess.run(
            [sys.executable, str(script), str(precision), str(columns), str(int(extra))],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return out.stdout.split()

    assert run(6, 0, False) == ["True", "0"]
    # Another max_columns (a notebook kernel's default) and an option
    # registered by an import: neither is read.
    assert run(6, 20, True) == ["True", "2"]
    assert run(2, 20, True) == ["True", "0"]
    assert run(2, 0, False) == ["True", "2"]
