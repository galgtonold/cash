"""Random values, random changes, and a check after every change that the
watched digest of every value is the digest a fresh hash gives.

Used by `test_watched_digests_match_a_fresh_hash` (a few hundred steps per
seed) and runnable on its own for long runs::

    PYTHONPATH=src python -m tests.test_core.watchers._fuzz --seconds 600
"""

from __future__ import annotations

import argparse
import copyreg
import gc
import random
import sys
import time
import types
from typing import Any

from cash import value_watch
from cash.value_hash import _compute_hash, compute_hash

_MOD_SOURCE = """
class A:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class B:
    kind = "b"
    def __init__(self, **kw):
        self.__dict__.update(kw)

class S:
    __slots__ = ("a", "b", "__weakref__")
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

class SD(S):
    __slots__ = ("c", "__dict__")

class Outer:
    class Inner:
        def __init__(self, v):
            self.v = v

def f(x):
    return x

def g(x):
    return -x
"""

_mods = [0]


def make_module() -> types.ModuleType:
    _mods[0] += 1
    name = f"_cwatch_fuzz_mod_{_mods[0]}"
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(_MOD_SOURCE, f"<{name}>", "exec"), mod.__dict__)
    return mod


class Fuzz:
    def __init__(self, seed: int, ns: dict) -> None:
        self.rng = random.Random(seed)
        self.mod = make_module()
        self.ns = ns
        self.pool: list = []  # containers and objects reachable from some root
        self.steps = 0
        self.reused = 0
        self.saved: dict = {}

    # -------------------------------------------------------- making values
    def atom(self) -> Any:
        r = self.rng
        return r.choice(
            [
                0,
                1,
                r.randint(-5, 300),
                10**30 + r.randint(0, 9),
                1.5,
                r.random(),
                "x",
                f"s{r.randint(0, 50)}",
                "ü" * r.randint(0, 3),
                b"by",
                None,
                True,
                False,
            ]
        )

    def value(self, depth: int = 0) -> Any:
        r = self.rng
        m = self.mod
        if depth > 3 or r.random() < 0.3:
            if self.pool and r.random() < 0.15:
                return r.choice(self.pool)  # an alias
            return self.atom()
        k = r.randrange(12)
        if k == 0:
            v: Any = {f"k{i}": self.value(depth + 1) for i in range(r.randint(0, 5))}
        elif k == 1:
            v = [self.value(depth + 1) for _ in range(r.randint(0, 5))]
        elif k == 2:
            v = {self.atom() for _ in range(r.randint(0, 5))}
        elif k == 3:
            v = tuple(self.value(depth + 1) for _ in range(r.randint(0, 4)))
        elif k == 4:
            v = frozenset(self.atom() for _ in range(r.randint(0, 4)))
        elif k == 5:
            v = m.A(x=self.value(depth + 1), y=self.atom())
        elif k == 6:
            v = m.B(z=self.value(depth + 1))
        elif k == 7:
            v = m.S(self.value(depth + 1), self.atom())
        elif k == 8:
            v = m.SD(self.atom(), self.value(depth + 1))
            v.c = self.value(depth + 1)
            v.extra = self.atom()
        elif k == 9:
            v = m.Outer.Inner(self.value(depth + 1))
        elif k == 10:
            v = r.choice([m.f, m.g, m.A, m.Outer.Inner])
            return v
        else:
            v = {i: [self.atom(), {"n": self.atom()}] for i in range(r.randint(0, 4))}
        if not isinstance(v, (tuple, frozenset)):
            self.pool.append(v)
        return v

    def new_root(self) -> None:
        name = f"v{self.rng.randint(0, 7)}"
        self.ns[name] = self.value()
        if self.rng.random() < 0.3:
            # roots that hold the same sub-objects
            other = f"v{self.rng.randint(0, 7)}"
            if other in self.ns and self.ns[other] is not self.ns[name]:
                self.ns[name] = [self.ns[other], self.value(2)]

    # -------------------------------------------------------- changing them
    def targets(self, kind: type | tuple) -> list:
        return [x for x in self.pool if isinstance(x, kind)]

    def mutate(self) -> str:
        r = self.rng
        m = self.mod
        k = r.randrange(30)
        if k < 3:
            self.new_root()
            return "new root"
        if k == 3:
            name = r.choice(list(self.ns) or ["v0"])
            if name in self.ns:
                v = self.ns[name]
                # an equal value built apart, in place of the old one
                self.ns[name] = _rebuild(v)
            return "rebind equal"
        if k == 4:
            for name in list(self.ns):
                if r.random() < 0.2:
                    del self.ns[name]
            self.pool = self.pool[-200:]
            gc.collect()
            return "drop"
        if k in (5, 6, 7):
            ds = self.targets(dict)
            if not ds:
                return "none"
            d = r.choice(ds)
            op = r.randrange(9)
            keys = list(d)
            if op == 0 or not keys:
                d[f"n{r.randint(0, 9)}"] = self.value(2)
            elif op == 1:
                del d[r.choice(keys)]
            elif op == 2:
                d[r.choice(keys)] = self.atom()
            elif op == 3:
                kk = r.choice(keys)
                d[kk] = d[kk]  # the same object again
            elif op == 4:
                kk = r.choice(keys)
                v = d.pop(kk)
                d[kk] = v  # moved to the end
            elif op == 5:
                d.clear()
            elif op == 6:
                d.update({f"u{i}": self.atom() for i in range(2)})
            elif op == 7:
                kk = r.choice(keys)
                v = d[kk]
                if type(v) is str:
                    d[kk] = "".join(list(v))  # equal, another object (or the same, interned)
            else:
                d.setdefault(f"sd{r.randint(0, 3)}", self.atom())
            return "dict"
        if k in (8, 9, 10):
            ls = [x for x in self.pool if type(x) is list]
            if not ls:
                return "none"
            lst = r.choice(ls)
            op = r.randrange(8)
            if op == 0:
                lst.append(self.value(2))
            elif op == 1 and lst:
                lst[r.randrange(len(lst))] = self.atom()
            elif op == 2 and lst:
                lst.pop(r.randrange(len(lst)))
            elif op == 3:
                lst.reverse()
            elif op == 4 and len(lst) > 1:
                i, j = r.randrange(len(lst)), r.randrange(len(lst))
                lst[i], lst[j] = lst[j], lst[i]
            elif op == 5:
                lst.insert(0, self.atom())
            elif op == 6 and lst:
                x = lst.pop()
                lst.append(x)  # same content again
            else:
                lst[:] = lst[:]
            return "list"
        if k == 11:
            ss = [x for x in self.pool if type(x) is set]
            if not ss:
                return "none"
            s = r.choice(ss)
            op = r.randrange(4)
            if op == 0:
                s.add(self.atom())
            elif op == 1 and s:
                s.discard(next(iter(s)))
            elif op == 2:
                x = self.atom()
                s.add(x)
                s.discard(x)
            else:
                items = list(s)
                s.clear()
                s.update(reversed(items))
            return "set"
        if k in (12, 13, 14):
            objs = self.targets((m.A, m.B, m.Outer.Inner))
            if not objs:
                return "none"
            o = r.choice(objs)
            op = r.randrange(7)
            if op == 0:
                o.x = self.value(2)
            elif op == 1 and vars(o):
                delattr(o, r.choice(list(vars(o))))
            elif op == 2:
                o.__dict__ = {"x": self.atom()}
            elif op == 3:
                o.__dict__.update(w=self.atom())
            elif op == 4 and type(o) in (m.A, m.B):
                o.__class__ = m.B if type(o) is m.A else m.A
            elif op == 5:
                setattr(o, "__getstate__", lambda: {"hidden": 1})
            else:
                o.__dict__.pop("__getstate__", None)
            return "object"
        if k == 15:
            objs = self.targets(m.S)
            if not objs:
                return "none"
            o = r.choice(objs)
            op = r.randrange(4)
            if op == 0:
                o.a = self.atom()
            elif op == 1:
                try:
                    del o.b
                except AttributeError:
                    o.b = 1
            elif op == 2 and isinstance(o, m.SD):
                o.c = self.value(2)
            else:
                o.b = self.value(2)
            return "slots"
        if k == 16:
            cls = r.choice([m.A, m.B, m.S, m.Outer.Inner])
            op = r.randrange(4)
            if op == 0:
                cls.tag = r.random()  # a class attribute: no pickle change, but watched
            elif op == 1:
                cls.__getstate__ = lambda self: {"from_class": 1}
            elif op == 2:
                if "__getstate__" in cls.__dict__:
                    del cls.__getstate__
            else:
                cls.__qualname__ = cls.__qualname__  # reassigned, same text
            return "class"
        if k == 17:
            op = r.randrange(6)
            if op == 0:
                self.saved.setdefault("A", m.A)
                m.A = r.choice([m.B, self.saved["A"]])  # global rebinding
            elif op == 1:
                if "A" in self.saved:
                    m.A = self.saved["A"]
            elif op == 2:
                self.saved.setdefault("f", m.f)
                m.f = r.choice([m.g, self.saved["f"]])
            elif op == 3:
                if "f" in self.saved:
                    m.f = self.saved["f"]
            elif op == 4:
                fn = r.choice([m.f, m.g])
                fn.__code__ = r.choice([m.f, m.g]).__code__
            else:
                saved_q = self.saved.setdefault("fq", m.g.__qualname__)
                m.g.__qualname__ = r.choice(["f", saved_q, "nowhere"])
            return "globals"
        if k == 18:
            op = r.randrange(3)
            if op == 0:
                copyreg.pickle(m.B, _reduce_b)
            else:
                copyreg.dispatch_table.pop(m.B, None)
            return "copyreg"
        if k == 19:
            # an old module name pointing at another module object
            name = m.__name__
            if r.random() < 0.5:
                sys.modules[name] = types.ModuleType(name)
            else:
                sys.modules[name] = m
            return "sys.modules"
        if k == 20:
            # roots rebuilt over and over: freed lists' addresses come back
            for i in range(r.randint(1, 4)):
                self.ns[f"r{i}"] = [str(r.randint(0, 3)), [self.atom()]]
            return "address reuse"
        # most steps: nothing changes, so the fast path gets exercised
        return "nothing"

    def check(self) -> None:
        for name, value in list(self.ns.items()):
            watched = compute_hash(value)
            fresh = _compute_hash(value)
            if watched != fresh:
                raise AssertionError(f"step {self.steps}: {name} watched {watched} != fresh {fresh}")
            # and once more, the second look is the one that reuses
            again = compute_hash(value)
            if again != fresh:
                raise AssertionError(f"step {self.steps}: {name} reused {again} != fresh {fresh}")

    def run(self, steps: int) -> list[str]:
        log = []
        for _ in range(steps):
            self.steps += 1
            what = self.mutate()
            log.append(what)
            self.check()
        return log

    def close(self) -> None:
        copyreg.dispatch_table.pop(self.mod.B, None)
        sys.modules.pop(self.mod.__name__, None)


def _reduce_b(obj: Any) -> tuple:
    return (object, ())


def _rebuild(v: Any) -> Any:
    if type(v) is dict:
        return {k: _rebuild(x) for k, x in v.items()}
    if type(v) is list:
        return [_rebuild(x) for x in v]
    if type(v) is set:
        return set(v)
    return v


def fuzz(seed: int, steps: int, ns: dict) -> Fuzz:
    """One run over *ns*, the namespace roots live in (an anchor)."""
    run = Fuzz(seed, ns)
    try:
        for _ in range(4):
            run.new_root()
        run.run(steps)
    finally:
        run.close()
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not value_watch.enabled():
        raise SystemExit(f"the fast path is off: {value_watch.REASON}")
    value_watch._cw.configure(min_bytes=0)
    ns: dict = {}
    main_ns = sys.modules["__main__"].__dict__
    value_watch._anchors = lambda: (ns, main_ns)
    t0 = time.time()
    seed = args.seed
    runs = 0
    while time.time() - t0 < args.seconds:
        ns.clear()
        fuzz(seed, args.steps, ns)
        seed += 1
        runs += 1
    s = value_watch.stats()
    print(f"{runs} runs x {args.steps} steps, seeds {args.seed}..{seed - 1}: OK")
    print({k: s[k] for k in ("checks", "hits", "arms", "commits", "verify_changed", "dict_events", "roots")})


if __name__ == "__main__":
    main()
