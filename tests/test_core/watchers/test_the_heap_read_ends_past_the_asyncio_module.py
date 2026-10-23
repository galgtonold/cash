"""The read of the heap for what holds an object ends, and finds what it
should, past the ``_asyncio`` module.

On Python 3.12.0-3.12.5 that module's own traverse can loop for ever once
its free list of future iterators comes back on itself, as it does after a
few ``asyncio.gather`` runs: ``gc.get_referents(sys.modules["_asyncio"])``
never returned, and a notebook statement whose outputs another variable
might share hung the kernel. The C read stops the module's traverse past
its state and reads its dict itself, losing nothing; the Python read cannot
stop it, so it answers "not known", which re-runs the statement with a
warning.
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import textwrap

import pytest

from cash import _native
from cash.notebook import shared_objects
from tests._cell_driver import run_cash_cell

_GATHERS = textwrap.dedent(
    """
    import asyncio, sys

    async def load():
        await asyncio.sleep(0)
        return 1

    async def main():
        for _ in range(5):
            await asyncio.gather(load(), load())

    for _ in range(3):
        asyncio.run(main())
    mod = sys.modules["_asyncio"]
    """
)


def _gathered() -> None:
    async def load():
        await asyncio.sleep(0)
        return 1

    async def main():
        for _ in range(5):
            await asyncio.gather(load(), load())

    for _ in range(3):
        asyncio.run(main())


needs_kernels = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")


@needs_kernels
def test_the_c_read_ends_and_finds_the_module_holding_its_dict():
    code = _GATHERS + textwrap.dedent(
        """
        from cash import _native
        found = _native.kernels.referrers_in([mod, [1]], {id(vars(mod))})
        print("found" if found == [mod] else f"missed {found!r}")
        """
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "found"


@needs_kernels
def test_the_c_read_ends_once_a_read_of_the_module_from_python_made_its_list_loop():
    """One ``gc.get_referents`` of the module from Python (a user's, a
    profiler's) frees the iterators of its list a second time, and on
    3.12.0-3.12.5 that makes the list come back on itself: the C read must end
    all the same."""
    code = (
        "from cash import _native\n"
        + _GATHERS
        + textwrap.dedent(
            """
        import gc, os
        r = gc.get_referents(mod)
        del r
        found = _native.kernels.referrers_in([mod, [1]], {id(vars(mod))})
        print("found" if found == [mod] else f"missed {found!r}", flush=True)
        # Python's own collection loops on the list from here on (an import,
        # the exit): not cash's to end.
        os._exit(0)
        """
        )
    )
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the C read did not end")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "found"


@needs_kernels
def test_the_c_read_finds_the_module_by_each_object_it_holds():
    _gathered()
    mod = sys.modules["_asyncio"]
    if shared_objects._TRAVERSE_LOOPS:
        held = [vars(mod)]  # what else it holds, Python cannot be asked
    else:
        held = gc.get_referents(mod)
    assert held
    for part in held:
        assert _native.kernels.referrers_in([mod], {id(part)}) == [mod]
    assert _native.kernels.referrers_in([mod], {id(object())}) == []


def test_the_python_read_is_not_known_where_the_module_s_read_can_loop(monkeypatch):
    _gathered()
    monkeypatch.setattr(shared_objects, "_REFERRERS_IN", None)
    monkeypatch.setattr(shared_objects, "_TRAVERSE_LOOPS", True)
    with pytest.raises(shared_objects.WalkBudgetExceeded):
        shared_objects._referrers_of(([1],), set())
    assert shared_objects.heap_unreadable()


def test_a_later_statement_does_not_inherit_a_read_that_could_not_be_done(cash_magics, monkeypatch):
    """A later search cut short for lack of time is not said to be one this
    Python cannot do."""
    monkeypatch.setattr(shared_objects, "_UNREADABLE", [True])
    run_cash_cell(cash_magics, "x = 1")
    assert not shared_objects.heap_unreadable()


def test_the_python_read_answers_where_the_module_s_read_ends(monkeypatch):
    if sys.version_info[:2] == (3, 12) and sys.version_info < (3, 12, 6):
        pytest.skip("the module's own read can loop for ever here")
    target = [1]
    holder = [target]
    monkeypatch.setattr(shared_objects, "_REFERRERS_IN", None)
    assert any(obj is holder for obj in shared_objects._referrers_of((target,), set()))


@needs_kernels
def test_a_read_for_few_objects_goes_through_the_c_read_where_gc_s_can_loop(monkeypatch):
    """``gc.get_referrers`` reads the module's traverse as well: where it can
    loop, the read for a few objects is the C read too."""
    _gathered()
    asked = []
    real = shared_objects._referrers_of
    monkeypatch.setattr(shared_objects, "_referrers_of", lambda objs, stop: asked.append(objs) or real(objs, stop))
    monkeypatch.setattr(shared_objects, "_TRAVERSE_LOOPS", True)
    target = [1]
    holder = [target]
    assert any(obj is holder for obj in shared_objects._referrers((target,), set()))
    assert asked == [(target,)]
