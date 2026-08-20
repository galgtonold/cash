"""The notebook claim for code-as-an-argument, against a REAL kernel.

``@cash.cache`` keyed its arguments by pickling them, and pickle serializes a
class or a function **by reference** -- module plus qualname, never its code.
So a notebook that passed a schema class, edited the class, and called again
got the previously cached answer back.

**Why this file exists at all, rather than an in-process harness.** The fix
hashes *bytecode*, not source, because ``inspect.getsource(cls)`` resolves a
class through ``sys.modules[cls.__module__].__file__`` and a kernel's
``__main__`` has none.

An in-process harness answers that question about *its own* process, and every
shape measured gives a different wrong answer:

* ``exec`` into a bare ``{}`` -- what ``tests/conftest.py``'s ``MockShell`` does
  -- leaves ``__module__ == 'builtins'``, and ``getsource`` raises
  ``TypeError: is a built-in class`` (3.10, 3.11, 3.14 alike).
* ``exec`` into a namespace named ``__main__`` inside a driver process resolves
  ``__main__`` to the **driver script**, and ``getsource`` opens and reads *that
  file*. On 3.10/3.11 it then searches it for the name and reports ``OSError:
  could not find class definition`` -- i.e. it failed only because that
  particular driver did not contain a matching definition. On 3.13+ there is no
  search: ``getsource`` goes by ``__firstlineno__``, which is 1 for an ``exec``'d
  class, so on 3.14 it **returns the driver's own source, unconditionally** --
  measured: the returned text was the driver file's own first line, not the
  class at all.
* ``InteractiveShell.instance()`` installs its own file-less ``__main__``, so it
  raises like a kernel -- and would NOT have false-passed. That is a reason to
  keep the real kernel rather than to relax: the conclusion would otherwise rest
  on an IPython implementation detail nothing pins.

None of those is the kernel's shape. (Interpreter labels matter here: this
suite's kernel is the interpreter running pytest, currently 3.11.)
``test_a_notebook_defined_class_has_no_source_but_is_still_user_code`` below
pins the kernel's shape by measurement, so a future rewrite onto
``inspect.getsource`` fails here with the reason attached rather than shipping.

**Why cash's notebook extension is OFF in these tests.** The claim under test
belongs to the ``@cash.cache`` decorator. With the extension on, editing the
class-definition cell also moves the *statement* cache's lineage, so the
calling statement re-executes on its own account, whether or not the
decorator's key moved. Measured with ``%cash_on``, both with the fold wired
and with both wiring lines deleted: the post-edit reading was
``RESULT == 'V2-EDITED'`` in **both** builds, and the recompute counter was
useless as an oracle either way (the statement cache restores the counter
list, so a repeat call reads ``len(RAN) == 0``). A cash-ON version of the test
below would pass against a build with the feature removed.
``assert_cash_active(False)`` is asserted rather than assumed, because ``cash
autoload`` can leave a "cash-off" kernel with caching still live.
"""
from __future__ import annotations

import pytest

# timeout(90): the global default is 30s and these three tests each boot a
# FRESH, unpooled kernel (with_cash=False opts out of warm reuse). 13.8s
# unloaded, but every residual flake in this suite has been a wall-clock
# measurement sitting near a threshold, so leave headroom for a loaded box.
pytestmark = [pytest.mark.integration, pytest.mark.core, pytest.mark.timeout(90)]


_V1 = "class Schema:\n    def render(self):\n        return 'V1'\n"
_V2 = "class Schema:\n    def render(self):\n        return 'V2-EDITED'\n"

# `RAN` is the recompute oracle; `RESULT` is the answer the user actually sees.
# Counting alone would not show the harm -- the bug's symptom is a *stale
# answer*, so both are asserted at the edit.
_DEFS = (
    "import cash\n"
    "RAN = []\n"
)
_FUNC = (
    "@cash.cache\n"
    "def render_with(schema):\n"
    "    RAN.append(1)\n"
    "    return schema().render()\n"
)
_CALL = "RESULT = render_with(Schema)\n"


def _bare_kernel(nb_runner, cells):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel(with_cash=False)
    # A control arm is worthless unless it really is one; see the fixture's
    # own docstring on `cash autoload`.
    nb_runner.assert_cash_active(False)
    return nb_runner


def test_a_notebook_defined_class_has_no_source_but_is_still_user_code(nb_runner):
    """The premise, measured in the environment it is a premise about.

    Three separate facts, each of which a source-based implementation needs
    and does not get:

    1. a kernel's ``__main__`` has no ``__file__``;
    2. ``inspect.getsource`` on a cell-defined class therefore raises;
    3. cash nevertheless classifies that class as user code and produces a
       code digest for it.

    (3) is the control for (1) and (2): without it this test would still pass
    against a build where the whole feature had been deleted, since "no source
    available" is true either way.
    """
    _bare_kernel(nb_runner, [
        "import sys, inspect, cash",
        _V1,
        "MAIN_FILE = getattr(sys.modules['__main__'], '__file__', None)\n"
        "try:\n"
        "    inspect.getsource(Schema)\n"
        "    SOURCE_ERR = None\n"
        "except Exception as exc:\n"
        "    SOURCE_ERR = type(exc).__name__\n"
        "_c = cash.Cash()\n"
        "IS_USER_CODE = _c._is_user_code_object(Schema)\n"
        "DIGEST = _c._code_surface_hash(Schema)\n",
    ]).run_all()

    assert nb_runner.peek("MAIN_FILE") == "None", (
        "a real kernel's __main__ is supposed to be file-less; if this ever "
        "reports a path, the harness is not a real kernel"
    )
    assert nb_runner.peek("SOURCE_ERR") == "'OSError'", (
        "inspect.getsource unexpectedly succeeded on a cell-defined class -- "
        "if that became possible, the bytecode primitive could be revisited"
    )
    assert nb_runner.peek("IS_USER_CODE") == "True", (
        "a cell-defined class must count as user code, or the fold never runs"
    )
    assert nb_runner.peek("DIGEST") != "None", (
        "control: cash must produce a code digest for the very class whose "
        "source is unreadable -- this is what bytecode buys"
    )


def test_editing_a_notebook_defined_class_invalidates_the_decorator_cache(nb_runner):
    """The motivating bug, end to end, in a real kernel.

    Two controls run before the claim, and both are load-bearing:

    * **the cache works at all** -- a repeat call must not recompute. An
      implementation that simply never caches (an unpicklable argument, a key
      error swallowed into a MISS) satisfies "the edit invalidated" for
      entirely the wrong reason.
    * **an identical re-definition still hits** -- re-running an unedited cell
      mints a brand-new class *object* at a new address. If the digest were
      keyed on identity rather than content, every re-run of any cell would
      invalidate, which is a cache that never hits dressed up as a fix.
    """
    runner = _bare_kernel(nb_runner, [_DEFS, _V1, _FUNC, _CALL])
    runner.run_all()

    assert runner.peek("len(RAN)") == "1", "the first call must actually run"
    assert runner.peek("RESULT") == "'V1'"

    runner.run_cell(4)
    assert runner.peek("len(RAN)") == "1", (
        "control: a repeat call with an untouched class must HIT. If this "
        "fails, nothing below distinguishes a fix from a cache that is simply "
        "broken."
    )

    runner.run_cell(2)          # re-run the class cell, source unchanged
    runner.run_cell(4)
    assert runner.peek("len(RAN)") == "1", (
        "control: re-defining an IDENTICAL class must still HIT. A digest "
        "keyed on the class object's identity rather than its code would "
        "invalidate here."
    )

    runner.set_cell_source(2, _V2)
    runner.run_cell(2)
    runner.run_cell(4)
    assert runner.peek("len(RAN)") == "2", (
        "editing a notebook-defined class did not invalidate -- the "
        "source-based path is back, or the user-code gate rejected __main__"
    )
    assert runner.peek("RESULT") == "'V2-EDITED'", (
        "the call recomputed but the answer is stale, which is the harm this "
        "feature exists to prevent"
    )


def test_an_opaque_notebook_class_does_not_invalidate(nb_runner):
    """``@cash.opaque`` opts a cell-defined class out, in a real kernel.

    The unmarked twin in the same kernel is the control: it takes the
    identical edit and must invalidate. Without it, an implementation that
    folded nothing at all would pass the opaque assertion.
    """
    marked_v1 = "@cash.opaque\nclass Marker:\n    def render(self):\n        return 'M1'\n"
    marked_v2 = "@cash.opaque\nclass Marker:\n    def render(self):\n        return 'M2'\n"
    plain_v1 = "class Plain:\n    def render(self):\n        return 'P1'\n"
    plain_v2 = "class Plain:\n    def render(self):\n        return 'P2'\n"

    runner = _bare_kernel(nb_runner, [
        _DEFS,
        marked_v1,
        plain_v1,
        "@cash.cache\n"
        "def takes(schema):\n"
        "    RAN.append(1)\n"
        "    return len(RAN)\n",
        "M = takes(Marker)\n",
        "P = takes(Plain)\n",
    ])
    runner.run_all()
    assert runner.peek("len(RAN)") == "2", "both first calls must run"

    runner.set_cell_source(2, marked_v2)
    runner.run_cell(2)
    runner.run_cell(5)
    assert runner.peek("len(RAN)") == "2", (
        "an @cash.opaque class must not contribute its code to the key"
    )

    runner.set_cell_source(3, plain_v2)
    runner.run_cell(3)
    runner.run_cell(6)
    assert runner.peek("len(RAN)") == "3", (
        "control: the unmarked twin took the same edit and must invalidate -- "
        "otherwise the assertion above holds because nothing is folded"
    )
