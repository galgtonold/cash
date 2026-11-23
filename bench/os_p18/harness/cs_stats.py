"""Cheap work counters for cash-mode replay kernels (correctness-set harness).

replay.py copies this file into the kernel's hook folder and, after ``%cash_on``,
calls ``install()`` through a silent user expression; after each cell it reads
``take()`` the same way. Nothing here changes what cash computes: each patch
wraps a function, counts the call (and bytes where that is cheap), times it,
and calls the original.

Counted (only calls made from cash's own code, unless the target is a cash
function itself):

  hash_bytes        bytes fed to cash's bulk digest (leaf hashes; values cash
                    hashes directly without the bulk digest are not counted)
  frame_copy        cash's own DataFrame / value copies before storing
                    (memory_backend._copy_frame, _safe_deep_copy;
                    frame_sharing._copy_values)
  deepcopy          copy.deepcopy called from cash
  pickle            pickle.dumps / pickle.dump / Pickler.dump called from cash
                    (bytes counted for dumps)
  referrer_scan     gc.get_referrers called from cash
  source_parse      ast.parse called from cash; source_read: inspect.getsource*
  store / restore   cash's statement store and hit restore (seconds)

``take()`` returns {counter: [calls, seconds, bytes]} since the last take plus
``_own``: the seconds the counters themselves added (two clock reads and a frame
check per call), and resets. Set CS_STATS=0 to turn the counters off.
"""

import os
import sys
import time

_perf = time.perf_counter
_C = {}
_own = [0.0]
_installed = []


def _bump(name, dt, nbytes=0):
    c = _C.get(name)
    if c is None:
        _C[name] = [1, dt, nbytes]
    else:
        c[0] += 1
        c[1] += dt
        c[2] += nbytes


def _from_cash(depth=2):
    try:
        return sys._getframe(depth).f_globals.get("__name__", "").startswith("cash")
    except ValueError:
        return False


def _wrap(owner, attr, name, filt=True, size=None, drop_args=False):
    raw = owner.__dict__.get(attr) if isinstance(owner, type) else None
    static = isinstance(raw, staticmethod)
    orig = getattr(owner, attr, None)
    if orig is None or getattr(orig, "_cs_stats", False):
        return
    def wrapped(*a, **k):
        t0 = _perf()
        if filt and not _from_cash():
            _own[0] += _perf() - t0
            r = orig(*a, **k)
            return [x for x in r if x is not a] if drop_args else r
        t1 = _perf()
        try:
            r = orig(*a, **k)
        finally:
            t2 = _perf()
        if drop_args:
            # the wrapper's own argument tuple holds the targets: gc.get_referrers
            # finds it, and cash never saw it, so take it out of the answer
            r = [x for x in r if x is not a]
        n = 0
        if size is not None:
            try:
                n = size(a, k, r)
            except Exception:  # noqa: BLE001
                n = 0
        _bump(name, t2 - t1, n)
        _own[0] += (t1 - t0) + (_perf() - t2)
        return r
    wrapped._cs_stats = True
    wrapped.__wrapped__ = orig
    try:
        wrapped.__name__ = getattr(orig, "__name__", attr)
        wrapped.__qualname__ = getattr(orig, "__qualname__", attr)
        wrapped.__doc__ = getattr(orig, "__doc__", None)
    except Exception:  # noqa: BLE001
        pass
    setattr(owner, attr, staticmethod(wrapped) if static else wrapped)
    _installed.append(f"{getattr(owner, '__name__', owner)}.{attr}")


def _len_of(i):
    def f(a, k, r):
        x = a[i] if len(a) > i else None
        try:
            return memoryview(x).nbytes
        except TypeError:
            return len(x) if hasattr(x, "__len__") else 0
    return f


def install():
    if os.environ.get("CS_STATS", "1") == "0":
        return "off"
    if _installed:
        return "already"
    import ast
    import copy
    import gc
    import inspect
    import pickle

    _wrap(gc, "get_referrers", "referrer_scan", drop_args=True)
    _wrap(ast, "parse", "source_parse")
    _wrap(inspect, "getsource", "source_read")
    _wrap(inspect, "getsourcelines", "source_read")
    _wrap(copy, "deepcopy", "deepcopy")
    _wrap(pickle, "dumps", "pickle", size=lambda a, k, r: len(r))
    _wrap(pickle, "dump", "pickle")
    try:
        base = pickle.Pickler

        class _CountingPickler(base):
            def dump(self, obj):
                if not _from_cash(2):
                    return super().dump(obj)
                t0 = _perf()
                try:
                    return super().dump(obj)
                finally:
                    _bump("pickle", _perf() - t0)
        _CountingPickler.__name__ = base.__name__
        _CountingPickler.__qualname__ = base.__qualname__
        _CountingPickler.__module__ = base.__module__
        pickle.Pickler = _CountingPickler
        _installed.append("pickle.Pickler.dump")
    except Exception:  # noqa: BLE001
        pass
    try:
        import cash.bulk_digest as bd
        # every leaf of the bulk digest goes through leaf_hasher(data) (blake3 hashes in the constructor)
        _wrap(bd, "leaf_hasher", "hash_bytes", filt=False, size=_len_of(0))
    except Exception:  # noqa: BLE001
        pass
    for mod, cls, attr, name in (
        ("cash.backends.memory_backend", "InMemoryBackend", "_copy_frame", "frame_copy"),
        ("cash.backends.memory_backend", "InMemoryBackend", "_safe_deep_copy", "value_copy"),
        ("cash.backends.frame_sharing", None, "_copy_values", "frame_copy_values"),
        ("cash.notebook.statement.store", "StatementStore", "save", "store"),
        ("cash.notebook.statement.hit", "CacheHitServer", "serve", "restore"),
    ):
        try:
            m = __import__(mod, fromlist=["_"])
            owner = getattr(m, cls) if cls else m
            if cls is None and not hasattr(owner, attr):
                continue
            _wrap(owner, attr, name, filt=False)
        except Exception:  # noqa: BLE001
            pass
    return ",".join(_installed)


def take():
    out = {k: [v[0], round(v[1], 6), v[2]] for k, v in _C.items()}
    out["_own"] = round(_own[0], 6)
    _C.clear()
    _own[0] = 0.0
    return out
