"""The library shape the callee-global defect needs: real modules, real edges.

Cached functions defined INSIDE a test function do not reproduce it. The call
edge `outer -> inner` is found by reading the source of a module, and a nested
def's callee is a closure name the graph never records -- so `outer` has no
dependencies at all and the fold under test is never reached. A fixture like
that passes against the unfixed code, which is the whole trap.

`_callee_rules` is a third module on purpose: the constant has to live outside
the caller's own module, or a channel that already worked would carry it.
"""
import cash
from cash import InMemoryBackend

from . import _callee_rules
from ._callee_rules import keep

c = cash.Cash(backend=InMemoryBackend(), register_magic=False)

#: Appended to by every body below, so a test can count executions rather than
#: infer them from a value that may be right for the wrong reason.
RUNS: list[str] = []


@c.cache(assume_safe=True)
def inner(n):
    """Reads the cross-module constant through a plain helper."""
    RUNS.append("inner")
    return sum(i for i in range(n) if keep(i))


@c.cache(assume_safe=True)
def outer(n):
    """The reported shape: a cached function calling a cached one."""
    RUNS.append("outer")
    return inner(n)


@c.cache(assume_safe=True)
def mid(n):
    RUNS.append("mid")
    return inner(n)


@c.cache(assume_safe=True)
def outer_deep(n):
    """Two cached levels above the constant."""
    RUNS.append("outer_deep")
    return mid(n)


@c.cache(assume_safe=True)
def inner_direct(n):
    """Reads the constant itself, with no helper in between."""
    RUNS.append("inner_direct")
    return sum(i for i in range(n) if (i % 100) < _callee_rules.THRESHOLD)


@c.cache(assume_safe=True)
def outer_direct(n):
    RUNS.append("outer_direct")
    return inner_direct(n)


@c.cache(assume_safe=True)
def unrelated(n):
    """No path to the rules module at all."""
    RUNS.append("unrelated")
    return n * 3
