"""A value written by hand into an Enum member's ``cached_property`` slot is
part of what the key reads.

What the getter stores on the member during a call stays out of the key --
its code stands for it, and the first read inside a call fills it -- but
``Color.RED.label = 99``, ``Color.RED.__dict__["label"] = 99`` or
``object.__setattr__(Color.RED, "label", 99)`` after the first call served
the result from before, through the global, a helper, a method and the
member handed as an argument. So did a value the getter stored outside any
call (the property read once before the first call) and then overwritten.
The enum lives in a module of the user's own package too.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

COLORS = textwrap.dedent("""
    import enum
    import functools


    class Color(enum.Enum):
        RED = 1
        BLUE = 2

        @functools.cached_property
        def label(self):
            return self.value * 10

        def bump(self, n):
            return self.label + n
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from colors import Color

    if "warm" in sys.argv:
        Color.RED.label


    @cash.cache
    def direct(n):
        return Color.RED.label + n


    @cash.cache
    def via_method(n):
        return Color.RED.bump(n)


    @cash.cache
    def via_argument(member, n):
        return member.label + n


    def show():
        return [[direct(1), via_method(1), via_argument(Color.RED, 1)] for _ in range(3)]


    print(show())
    Color.RED.label = 99
    print(show())
    Color.RED.__dict__["label"] = 199
    print(show())
    object.__setattr__(Color.RED, "label", 299)
    print(show())
    del Color.RED.label
    print(show())
""")


@pytest.mark.parametrize("argv", [(), ("warm",)], ids=["cold", "warmed"])
def test_a_hand_written_slot_matches_plain_python(tmp_path, argv):
    (tmp_path / "colors.py").write_text(COLORS, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    plain = run_python("main.py", *argv, cwd=tmp_path, env={"CASH_DISABLE": "1"}).stdout
    cached = run_python("main.py", *argv, cwd=tmp_path).stdout
    assert cached == plain, f"cash printed\n{cached}\nplain Python printed\n{plain}"
    again = run_python("main.py", *argv, cwd=tmp_path).stdout
    assert again == plain, f"a second process printed\n{again}\nplain Python printed\n{plain}"
