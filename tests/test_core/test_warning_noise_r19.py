"""Warnings that were wrong, so people learned to skip the right ones (round 19).

* ``np.sort(x)`` reported as a "write method": a module matched ``sort``.
* ``time.sleep(...)`` and a call to the module's own log helper reported as
  ``discarded_call`` -- the quickstart's example tripped it.
* A print inside a ``functools.wraps`` wrapper numbered from the WRAPPED
  function's first line, in another file.
* KEY-UNHASHABLE-GLOBAL for ``Model.fit.ACTIVE_CONFIG`` on any class with a
  cached method: the class walk read cash's own wrapper.
* ``mutable_global`` on the loader of a lazily filled settings dict.
"""
from __future__ import annotations

import importlib
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


def _module(tmp_path, monkeypatch, files: dict[str, str]):
    tag = f"_{time.monotonic_ns()}"
    for name, text in files.items():
        (tmp_path / f"{name}{tag}.py").write_text(
            textwrap.dedent(text).replace("{tag}", tag), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    mods = {name: importlib.import_module(f"{name}{tag}") for name in files}
    for full in [f"{n}{tag}" for n in files]:
        monkeypatch.setitem(sys.modules, full, sys.modules[full])
    return mods


def _messages(tmp_path, fn, *args):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(fn)(*args)
    return "\n".join(str(w.message) for w in rec)


def test_a_numpy_function_named_like_a_list_method_is_not_a_write(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    mods = _module(tmp_path, monkeypatch, {"npsort": """
        import numpy as np

        def work(x):
            y = np.sort(x)
            z = np.append(y, 1.0)
            return float(z[0])
    """})
    text = _messages(tmp_path, mods["npsort"].work, np.array([2.0, 1.0]))
    assert "write method" not in text, text


def test_a_numpy_save_is_still_a_write(tmp_path, monkeypatch):
    """Control: only the container-mutator names are exempt on a module."""
    np = pytest.importorskip("numpy")
    out = tmp_path / "out.npy"
    mods = _module(tmp_path, monkeypatch, {"npsave": f"""
        import numpy as np

        def work(x):
            np.save({str(out)!r}, x)
            return 1
    """})
    assert "np.save() - write method" in _messages(tmp_path, mods["npsave"].work, np.array([1.0]))


def test_sleep_and_a_log_helper_are_not_discarded_calls(tmp_path, monkeypatch):
    mods = _module(tmp_path, monkeypatch, {"sleepy": """
        import sys, time
        from time import sleep

        def _log(msg):
            print(msg, file=sys.stderr)

        def work(n):
            time.sleep(0.001)
            sleep(0.001)
            _log("step")
            return n
    """})
    text = _messages(tmp_path, mods["sleepy"].work, 1)
    assert "discards return of time.sleep" not in text
    assert "discards return of sleep" not in text
    assert "discards return of _log" not in text
    assert "discards return of print" not in text, "print was reported twice"


def test_a_line_inside_a_wraps_wrapper_is_numbered_in_its_own_file(tmp_path, monkeypatch):
    mods = _module(tmp_path, monkeypatch, {
        "deco": """
            import functools, sys

            def timed(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    out = fn(*args, **kwargs)
                    print("[timed]", file=sys.stderr)  # line 8
                    return out
                return wrapper
        """,
        "helpers": "\n" * 40 + """
from deco{tag} import timed

@timed
def slow_square(x):
    return x * x

def work(x):
    return slow_square(x)
""",
    })
    text = _messages(tmp_path, mods["helpers"].work, 3)
    assert "line 8: [impure_call] print()" in text, text


def test_a_class_with_cached_methods_does_not_report_cash_s_own_globals(tmp_path, monkeypatch):
    mods = _module(tmp_path, monkeypatch, {"model": """
        import cash

        c = cash.Cash(cache_dir={cache!r})

        class Model:
            def __init__(self, k):
                self.k = k

            @c.cache
            def fit(self, xs):
                return [x * self.k for x in xs]

            @c.cache
            def score(self, xs):
                return sum(xs) * self.k
    """.replace("{cache!r}", repr(str(tmp_path / "mcache")))})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mods["model"].Model(3).fit([1, 2])
    assert not [w for w in rec if "ACTIVE_CONFIG" in str(w.message)]


def test_the_loader_of_a_settings_dict_is_not_a_stale_read(tmp_path, monkeypatch):
    mods = _module(tmp_path, monkeypatch, {"settings": """
        _CFG = {}

        def load():
            _CFG.clear()
            _CFG.update({"threshold": 3})
            return _CFG

        def get(name):
            if not _CFG:
                load()
            return _CFG.get(name)

        def work():
            return get("threshold") * 10
    """})
    mods["settings"].load()
    text = _messages(tmp_path, mods["settings"].work)
    assert "mutable_global" not in text, text
