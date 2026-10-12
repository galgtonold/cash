"""A global the key read, changed by another thread while the call runs,
does not become a wrong entry.

The key is built before the body runs. A config reload in another thread
(``RATE = 2``, ``CFG["fee"] = 5``, ``Settings.scale = 3``) landing mid-call
made the body compute from the new values, and the result was stored under
the key of the old ones: every later call with the old values, in this
process and the next, was served it. Now the program state the key read is
checked again before the store when other threads are running, and a result
whose inputs moved is returned but not cached, with STORE-STATE-CHANGED.

The writer is a thread sequenced with Events, not a sleep race.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import threading
import warnings

import pytest

pytestmark = pytest.mark.core

MODULE = textwrap.dedent("""
    import threading

    RATE = 1
    CFG = {"fee": 1}
    FEES = [1]

    class Settings:
        scale = 1

    #: Sync objects are left out of the key.
    STARTED = threading.Event()
    RELOADED = threading.Event()

    def _pause():
        STARTED.set()
        RELOADED.wait(10)

    def price(x):
        _pause()
        return x * RATE * Settings.scale + CFG["fee"]

    def total(x):
        _pause()
        return x + sum(FEES)
""")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    name = f"svc_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    mod = importlib.import_module(name)
    yield mod
    sys.modules.pop(name, None)


def _call_while_reloading(fn, reload, undo):
    """Run fn(10) in a thread; reload mid-call; undo once it returned."""
    mod = sys.modules[fn.__wrapped__.__module__]
    mod.STARTED.clear()
    mod.RELOADED.clear()
    out = []
    t = threading.Thread(target=lambda: out.append(fn(10)))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        t.start()
        assert mod.STARTED.wait(10)
        reload(mod)
        mod.RELOADED.set()
        t.join(10)
    undo(mod)
    mod.RELOADED.set()  # later calls do not wait
    return out[0], [str(w.message) for w in rec]


def _rebind(mod):
    mod.RATE = 2


def _rebind_back(mod):
    mod.RATE = 1


def _in_place(mod):
    mod.CFG["fee"] = 5


def _in_place_back(mod):
    mod.CFG["fee"] = 1


def _class_attr(mod):
    mod.Settings.scale = 3


def _class_attr_back(mod):
    mod.Settings.scale = 1


@pytest.mark.parametrize(
    ("reload", "undo", "during"),
    [(_rebind, _rebind_back, 21), (_in_place, _in_place_back, 15), (_class_attr, _class_attr_back, 31)],
    ids=["rebound", "changed_in_place", "class_attribute"],
)
def test_the_result_of_the_new_value_is_not_stored_under_the_old_key(disk_cash, svc, reload, undo, during):
    price = disk_cash.cache(svc.price)
    got, messages = _call_while_reloading(price, reload, undo)
    assert got == during, "the call returns what the body computed"
    assert any("STORE-STATE-CHANGED" in m for m in messages), messages
    assert price(10) == 11
    assert price(10) == 11 and price.cache_info()["hits"] == 1, "the settled value caches normally"


def test_a_global_passed_to_a_call_and_changed_by_a_thread_is_not_stored(disk_cash, svc):
    """``sum(FEES)``: a global passed to a call is watched for the call
    itself changing it; another thread changing it is not the call's doing,
    and its result is not stored under the old key."""
    total = disk_cash.cache(svc.total)

    def change(mod):
        mod.FEES[0] = 5

    def back(mod):
        mod.FEES[0] = 1

    got, messages = _call_while_reloading(total, change, back)
    assert got == 15
    assert any("STORE-STATE-CHANGED" in m for m in messages), messages
    assert total(10) == total.__wrapped__(10) == 11
