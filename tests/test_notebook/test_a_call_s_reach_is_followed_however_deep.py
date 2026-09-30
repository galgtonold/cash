"""The globals a call can read are followed through every function it reaches.

`global_names_reached` decides which loop-written names a call in a loop
unit can see, and which loop variables a content-keyed call may leave out of
its key. It followed six calls: a name read seven functions down was not
seen, and the call was taken as if it could not read it.
"""

from __future__ import annotations

import types

from cash.notebook.call_key import global_names_reached


def _chain(length: int) -> types.FunctionType:
    namespace: dict = {}
    source = [f"def f{i}():\n    return f{i + 1}()\n" for i in range(length - 1)]
    source.append(f"def f{length - 1}():\n    return FACTOR\n")
    exec("\n".join(source), namespace)  # noqa: S102 - test-built source
    return namespace["f0"]


def test_a_global_read_twelve_calls_down_is_reached_and_a_cycle_ends():
    assert "FACTOR" in global_names_reached(_chain(12))
    namespace: dict = {}
    exec("def a():\n    return b()\n\ndef b():\n    return a() + FACTOR\n", namespace)  # noqa: S102 - test-built source
    assert {"b", "a", "FACTOR"} <= global_names_reached(namespace["a"])
