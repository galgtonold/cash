"""The seed functions cash watches still pickle, and a disabled cash
replaces none of them.

From ``import cash`` on, ``random.seed``, ``numpy.random.seed`` and the
functions that read or set a global stream's state are cash's: a seed is
part of a cached call's key. Programs hand those functions to process pools
(``pool.map(np.random.seed, seeds)``, ``Pool(initializer=random.seed,
initargs=(0,))``), and the wrappers could not be pickled: ``AttributeError:
Can't get local object`` as soon as cash was imported anywhere in the
program, also with ``CASH_DISABLE=1``.
"""

from __future__ import annotations

import os
import pickle
import random
import textwrap

import pytest

import cash
from tests._scripts import run_python

pytestmark = pytest.mark.core


def test_the_seed_functions_pickle_after_import_cash():
    import numpy as np

    assert cash  # imported: the seed watch is on
    for fn in (
        random.seed,
        random.getstate,
        random.setstate,
        np.random.seed,
        np.random.get_state,
        np.random.set_state,
    ):
        assert getattr(fn, "__cash_seed_watch__", False), f"{fn!r} is not watched: nothing is tested"
        loaded = pickle.loads(pickle.dumps(fn))
        assert callable(loaded)
        assert loaded.__name__ == fn.__name__ == fn.__wrapped__.__name__
        assert fn.__module__ == fn.__wrapped__.__module__
        assert fn.__qualname__ == fn.__wrapped__.__qualname__
        assert isinstance(fn, type(fn.__wrapped__))
    # A bound method pickles by its receiver, as plain Python does it: the
    # loaded function seeds a copy of the stream.
    loaded = pickle.loads(pickle.dumps(random.seed))
    assert isinstance(loaded.__self__, random.Random) and loaded.__self__ is not random.seed.__self__


_SEED_POOL = """
    import multiprocessing as mp, pickle, random, sys
    import numpy as np

    def view():
        out = []
        for fn in (random.seed, np.random.seed, random.getstate, np.random.get_state):
            try:
                pickle.dumps(fn); out.append("pickles")
            except Exception as e:
                out.append(type(e).__name__)
        for method in ("fork", "spawn"):
            try:
                with mp.get_context(method).Pool(2) as pool:
                    pool.map(np.random.seed, [1, 2])
                with mp.get_context(method).Pool(1, initializer=random.seed, initargs=(0,)) as pool:
                    pool.map(abs, [1])
                out.append("pool works")
            except Exception as e:
                out.append(f"{type(e).__name__}: {e}")
        return out

    if __name__ == "__main__":
        plain = view()
        import cash
        watched = view()
        print("REPLACED", random.seed.__class__.__name__, type(random.seed).__name__)
        print("SAME" if plain == watched else f"DIFFERENT {plain} {watched}")
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
@pytest.mark.timeout(120)
def test_a_pool_is_seeded_with_the_seed_functions_after_import_cash(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(_SEED_POOL), encoding="utf-8")
    out = run_python(script, cwd=tmp_path).stdout
    assert "REPLACED method StandIn" in out, out  # the watch was on
    assert "SAME" in out, out


_DISABLED = """
    import pickle, random
    import numpy as np
    seeds = (random.seed, random.getstate, random.setstate, np.random.seed, np.random.get_state, np.random.set_state)
    import cash

    @cash.cache
    def f(x):
        return x + np.random.rand()

    f(1)
    now = (random.seed, random.getstate, random.setstate, np.random.seed, np.random.get_state, np.random.set_state)
    print("UNTOUCHED" if all(a is b for a, b in zip(seeds, now)) else "REPLACED")
"""


def test_a_disabled_cash_replaces_no_seed_function(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(_DISABLED), encoding="utf-8")
    assert "UNTOUCHED" in run_python(script, cwd=tmp_path, env={"CASH_DISABLE": "1"}).stdout
    # Positive control: the same script with caching on is watched.
    assert "REPLACED" in run_python(script, cwd=tmp_path).stdout
