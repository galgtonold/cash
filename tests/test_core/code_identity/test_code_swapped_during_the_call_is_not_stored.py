"""A call during which code its key folded was replaced is not stored.

Some decorators compile lazily: networkx's ``argmap`` returns a stub whose
code only calls the decorator, and the first call sets the stub's
``__code__`` to generated code that runs the decorated body through a
global of its own. The first call's key was built on the stub and left out
what the body reads; stored, its entry was found by every later process
that starts with the same stub, after the body had changed. That call is
now returned but not stored, and the next one keys the code that ran.

The decorator below does what ``argmap`` does, with nothing of networkx.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from cash import Cash

pytestmark = pytest.mark.core

#: A lazily compiling decorator. The body is held only through a weak
#: reference, so before the first call nothing the key can follow leads to
#: it: only the code the first call puts in place does.
LAZY = """
import weakref


def _template(*args, **kwargs):
    return _TARGET_(*args, **kwargs)


def _again(*args, **kwargs):
    return _TARGET_(*args, **kwargs)


_KEEP = []


def lazy(fn):
    def stub(*args, _lazy_self=None, **kwargs):
        return lazy_compile(_lazy_self)(*args, **kwargs)

    stub.__name__ = fn.__name__
    stub.__qualname__ = fn.__qualname__
    stub.__module__ = fn.__module__
    stub.__kwdefaults__ = {"_lazy_self": stub}
    stub.lazy_target = weakref.ref(fn)
    _KEEP.append(fn)
    return stub


def lazy_compile(stub):
    fn = stub.lazy_target()
    name = "lazy_" + fn.__name__
    stub.__globals__[name] = fn  # @cash:assume-safe
    code = _template.__code__
    names = tuple(name if n == "_TARGET_" else n for n in code.co_names)
    stub.__code__ = code.replace(co_names=names, co_filename="<lazy-gen>", co_name=fn.__name__)  # @cash:assume-safe
    return stub


def recompile(fn):
    \"\"\"A wrapper that sets *fn*'s code anew on every call.\"\"\"

    def wrapper(*args):
        source = _again if fn.__code__.co_name != "_again" else _template
        fn.__code__ = source.__code__.replace(co_names=fn.__code__.co_names)  # @cash:assume-safe
        return fn(*args)

    return wrapper
"""

USER = """
from {lazy} import lazy

RATE = 2


@lazy
def scaled(x):
    return x * RATE{extra}


def run(x):
    return scaled(x) + 1
"""


@pytest.fixture
def modules(tmp_path, monkeypatch):
    """``load(extra="")``: the lazy decorator's module and a user module
    whose ``run`` calls a lazily decorated ``scaled``, imported afresh, as a
    new process imports them."""
    stem = tmp_path.name.replace("-", "_")
    lazy_name, user_name = f"lazylib_{stem}", f"lazyuser_{stem}"
    (tmp_path / f"{lazy_name}.py").write_text(LAZY, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def load(extra: str = ""):
        for name in (user_name, lazy_name):
            sys.modules.pop(name, None)
        text = USER.format(lazy=lazy_name, extra=extra)
        (tmp_path / f"{user_name}.py").write_text(text, encoding="utf-8", newline="\n")
        importlib.invalidate_caches()
        return importlib.import_module(user_name)

    yield load
    for name in (user_name, lazy_name):
        sys.modules.pop(name, None)


def _outcome(cash: Cash) -> dict:
    return list(cash._misses.outcomes.values())[-1]


def _store_warnings(rec) -> list[str]:
    return [str(w.message) for w in rec if "[STORE-" in str(w.message)]


def test_the_call_that_compiles_the_code_is_not_stored(modules, cash_instance):
    mod = modules()
    run = cash_instance.cache(mod.run)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert run(3) == 7
        assert "replaced while it ran" in (_outcome(cash_instance).get("not_stored") or "")
        assert run(3) == 7
        assert "stored_at" in _outcome(cash_instance)
        assert run(3) == 7
    info = run.cache_info()
    assert (info["misses"], info["hits"]) == (2, 1)
    # Once is expected: said in the miss reason, not warned about.
    assert not _store_warnings(rec)


def test_the_stored_key_covers_what_the_compiled_body_reads(modules, cash_instance, monkeypatch):
    mod = modules()
    run = cash_instance.cache(mod.run)
    run(3)
    assert run(3) == 7
    monkeypatch.setattr(mod, "RATE", 5)
    assert run(3) == run.__wrapped__(3) == 16
    assert run.cache_info()["hits"] == 0


def test_a_new_process_is_not_served_a_body_it_no_longer_has(modules, clean_backend):
    """The first call's key is the same in every process that starts with
    the stub, whatever the body says: nothing may be stored under it. A new
    `Cash` over the same backend, with the modules imported afresh, is the
    next process; no other thread runs, so only this check stands."""
    run = Cash(backend=clean_backend, register_magic=False).cache(modules().run)
    assert run(3) == 7
    assert run(3) == 7

    # The next process: the decorated body changed, the stub did not.
    later = Cash(backend=clean_backend, register_magic=False)
    run = later.cache(modules(extra=" + 100").run)
    assert run(3) == 107
    assert run(3) == 107


def test_code_replaced_on_every_call_is_never_stored_and_says_why(modules, cash_instance):
    mod = modules()
    mod.scaled(1)
    mod.scaled = sys.modules[mod.lazy.__module__].recompile(mod.scaled)
    run = cash_instance.cache(mod.run)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(4):
            assert run(3) == 7
    assert run.cache_info()["hits"] == 0
    swapped = [m for m in _store_warnings(rec) if "STORE-CODE-SWAPPED" in m]
    assert len(swapped) == 1, _store_warnings(rec)
    assert "scaled" in swapped[0]


def test_code_that_stays_put_is_stored_on_the_first_call(modules, cash_instance):
    """Control: a call whose code was compiled before it is stored at once."""
    mod = modules()
    mod.scaled(1)  # compiled outside any cached call
    run = cash_instance.cache(mod.run)
    assert run(3) == 7
    assert run(3) == 7
    assert run.cache_info()["hits"] == 1
