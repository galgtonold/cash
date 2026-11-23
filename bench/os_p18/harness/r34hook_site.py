"""sitecustomize for plain-mode script steps of the round-34 replay harness.

replay.py copies this file to <tmp>/hook/sitecustomize.py and puts <tmp>/hook
first on PYTHONPATH, only for 'script' and 'shell' steps in plain mode
(CASH_DISABLE=1). When cash.core is imported, `Cash.cache` is patched so every
function it decorates is wrapped in a timer. Each call appends one record to
$R34_CALLS_DIR/calls-<pid>.jsonl at exit:

  {"fn": module.qualname, "q": qualname, "a": argument digest, "f": function
   identity, "t": seconds inside the call minus the hook's own time in nested
   calls, "d": nesting depth, "id": n, "p": parent id or -1, "ok": bool}

plus one {"overhead": seconds} line: the hook's own time, which replay.py
subtracts from the step's wall time. Nothing happens unless R34_CALLS_DIR is
set. Any other sitecustomize on sys.path still runs afterwards.
"""

import os
import sys


def _r34_install():
    calls_dir = os.environ.get("R34_CALLS_DIR")
    if not calls_dir:
        return
    import atexit
    import functools
    import importlib.machinery
    import inspect
    import json
    import time

    here = os.path.dirname(os.path.abspath(__file__))
    state = {"buf": [], "overhead": 0.0, "n": 0, "stack": [], "done": False, "stmt": None}
    sys._r34_calls = state  # the replay kernel reads and clears buf/overhead after each cell
    perf = time.perf_counter

    def flush():
        if not state["buf"] and state["done"]:
            return
        try:
            with open(os.path.join(calls_dir, f"calls-{os.getpid()}.jsonl"), "a") as fh:
                for r in state["buf"]:
                    fh.write(json.dumps(r) + "\n")
                fh.write(json.dumps({"overhead": round(state["overhead"], 6)}) + "\n")
            state["buf"].clear()
            state["overhead"] = 0.0
        except Exception:  # noqa: BLE001 - never break the user's script
            pass

    atexit.register(flush)

    _dig = {}

    def dig():
        if not _dig:
            sys.path.insert(0, here)
            try:
                import r34digest

                _dig["m"] = r34digest
            finally:
                try:
                    sys.path.remove(here)
                except ValueError:
                    pass
        return _dig["m"]

    def timing_wrap(w, func):
        name = f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', '?')}"
        qual = getattr(func, "__qualname__", "?")

        def before(a, k):
            t0 = perf()
            try:
                d = dig()
                ad = d.args_digest(a, k)
                fd = d.func_identity(func)
            except Exception as e:  # noqa: BLE001
                ad, fd = "ERR " + type(e).__name__, "?"
            state["n"] += 1
            cid = state["n"]
            parent = state["stack"][-1][0] if state["stack"] else -1
            state["stack"].append([cid, 0.0])
            t1 = perf()
            state["overhead"] += t1 - t0
            return ad, fd, cid, parent, t1, t1 - t0

        def after(ad, fd, cid, parent, t1, pre, ok):
            t2 = perf()
            _, child_over = state["stack"].pop()
            state["buf"].append(
                {"fn": name, "q": qual, "a": ad, "f": fd, "t": round(t2 - t1 - child_over, 6), "d": len(state["stack"]),
                 "id": cid, "p": parent, "ok": ok, "s": state["stmt"]}
            )
            if len(state["buf"]) >= 2000 and state["stmt"] is None:
                flush()
            t3 = perf()
            state["overhead"] += t3 - t2
            if state["stack"]:
                state["stack"][-1][1] += child_over + pre + (t3 - t2)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(w)
            async def tw(*a, **k):
                info = before(a, k)
                ok = False
                try:
                    r = await w(*a, **k)
                    ok = True
                    return r
                finally:
                    after(*info, ok)
        else:

            @functools.wraps(w)
            def tw(*a, **k):
                info = before(a, k)
                ok = False
                try:
                    r = w(*a, **k)
                    ok = True
                    return r
                finally:
                    after(*info, ok)

        tw.__wrapped__ = getattr(w, "__wrapped__", func)
        return tw

    def patch(core):
        Cash = getattr(core, "Cash", None)
        if Cash is None or getattr(Cash.cache, "_r34", False):
            return
        orig = Cash.cache

        @functools.wraps(orig)
        def cache(self, func=None, **kw):
            r = orig(self, func, **kw)
            if func is None or not callable(r):
                return r  # the decorator-with-options path calls self.cache(f, ...) again
            try:
                return timing_wrap(r, func)
            except Exception:  # noqa: BLE001
                return r

        cache._r34 = True
        Cash.cache = cache

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "cash.core":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    patch(module)
                except Exception:  # noqa: BLE001
                    pass

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Finder())


try:
    _r34_install()
except Exception:  # noqa: BLE001
    pass


def _r34_chain():
    """Run the next sitecustomize on sys.path (e.g. one a venv ships)."""
    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or ".") != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec and spec.origin and os.path.abspath(spec.origin) != os.path.abspath(__file__):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


try:
    _r34_chain()
except Exception:  # noqa: BLE001
    pass
