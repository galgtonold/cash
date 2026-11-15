"""A numpy ``Generator`` subclass that hands every draw to a ``RandomState``
it holds is cached like a plain generator argument.

scanpy wraps a legacy ``random_state`` this way: a ``Generator`` subclass
with no bit generator of its own, whose methods are made in a loop, each
``functools.wraps``-named after the ``Generator`` method it replaces and
forwarding with ``getattr(self.state, name)(...)``. Its stream is the held
``RandomState``: the key reads that state, a miss draws from it and a hit
moves it on to where the computed call left it. The delegates share one
code object, each closing over the name it forwards, so the trace digests
each by its own cells. Drawn from the module's own stream, it stays an
unseeded draw: not served.
"""

from __future__ import annotations

import sys
import textwrap
import types
from functools import wraps

import numpy as np
import pytest

from cash.decorator.user_code import is_user_code_object
from tests._scripts import run_python

WRAPPER = """
from functools import WRAPPER_ASSIGNMENTS, wraps

import numpy as np


class LegacyRng(np.random.Generator):
    def __init__(self, arg, state=None):
        self.arg = arg
        self.state = np.random.RandomState(arg) if state is None else state

    @property
    def bit_generator(self):
        raise AttributeError("no bit generator of its own")

    @classmethod
    def _delegate(cls):
        for name, meth in np.random.Generator.__dict__.items():
            if name.startswith("_") or not callable(meth) or name in cls.__dict__:
                continue

            def mk_wrapper(name, meth):
                @wraps(meth, assigned=set(WRAPPER_ASSIGNMENTS) - {"__doc__"})
                def wrapper(self, *args, **kwargs):
                    return getattr(self.state, name)(*args, **kwargs){scale}

                return wrapper

            setattr(cls, name, mk_wrapper(name, meth))


LegacyRng._delegate()
"""

SCRIPT = """
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import cash
from wrap import LegacyRng


@cash.cache
def draw(n, rng):
    return rng.normal(size=n)


rng = LegacyRng(int(sys.argv[1]))
print(draw.explain(3, rng).reason)
first = draw(3, rng)
second = draw(3, rng)
after = rng.normal()
print(" ".join(f"{v:.6f}" for v in [*first, *second, after]))
"""


def _write(tmp_path, scale: str = "") -> None:
    (tmp_path / "wrap.py").write_text(textwrap.dedent(WRAPPER.replace("{scale}", scale)), encoding="utf-8")
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")


def _run(tmp_path, seed: int) -> tuple[str, str]:
    reason, values = run_python("main.py", str(seed), cwd=tmp_path).stdout.strip().splitlines()[-2:]
    return reason, values


def _expected(seed: int, scale: float = 1.0) -> str:
    rs = np.random.RandomState(seed)
    values = [*(rs.normal(size=3) * scale), *(rs.normal(size=3) * scale), rs.normal() * scale]
    return " ".join(f"{v:.6f}" for v in values)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the delegates' run-time lookup is served on a trace (3.12+)")
def test_a_seeded_wrapper_hits_and_moves_the_callers_stream_on(tmp_path):
    _write(tmp_path)
    assert _run(tmp_path, 0) == ("no_entry", _expected(0))
    # Both calls served, and the caller's generator left where the computed
    # calls left it: the draw after them is the stream's seventh value.
    assert _run(tmp_path, 0) == ("hit", _expected(0))
    # Another seed is another stream.
    assert _run(tmp_path, 1) == ("no_entry", _expected(1))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the delegates' run-time lookup is served on a trace (3.12+)")
def test_an_edit_to_the_delegates_is_not_served_the_old_result(tmp_path):
    _write(tmp_path)
    assert _run(tmp_path, 0)[0] == "no_entry"
    assert _run(tmp_path, 0)[0] == "hit"

    _write(tmp_path, scale=" * 2")
    reason, values = _run(tmp_path, 0)
    assert reason != "hit"
    assert values == _expected(0, scale=2.0)


def test_a_wrapper_over_the_module_stream_is_not_served(cash_instance):
    """Holding the global ``RandomState``, it draws from the module stream:
    with no seed set in the call, each call draws anew."""
    ns: dict = {}
    exec(compile(textwrap.dedent(WRAPPER.replace("{scale}", "")), "<wrap>", "exec"), ns)
    rng = ns["LegacyRng"](None, np.random.mtrand._rand)

    @cash_instance.cache
    def draw(n, rng):
        return rng.normal(size=n)

    first = draw(3, rng)
    second = draw(3, rng)
    assert not np.array_equal(first, second)


def test_a_library_delegate_named_after_what_it_wraps_is_not_user_code():
    """``functools.wraps`` gives it numpy's module and qualified name, which
    finds the C method: its own code, in an installed file, is the
    library's."""
    ns: dict = {}
    source = "def wrapper(self, *args, **kwargs):\n    return getattr(self.state, name)(*args, **kwargs)\n"
    exec(compile(source, np.__file__, "exec"), ns)
    delegate = wraps(np.random.Generator.normal)(ns["wrapper"])
    assert isinstance(delegate, types.FunctionType)
    assert not is_user_code_object(delegate)

    # The same wraps over a function of the user's file stays the user's.
    exec(compile(source, __file__, "exec"), ns)
    assert is_user_code_object(wraps(np.random.Generator.normal)(ns["wrapper"]))
