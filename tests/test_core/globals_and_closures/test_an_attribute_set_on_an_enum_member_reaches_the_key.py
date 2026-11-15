"""An attribute an Enum member holds is part of what the key reads.

A member pickles by name, and the key read an Enum as its members' names and
values: ``Color.RED.extra = 20`` (or a write to ``Color.RED.__dict__``)
after a call that read ``Color.RED.extra`` served the result from before.
The script below runs each call several times (the memoised hit paths too)
and must print what plain Python prints.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

SCRIPT = textwrap.dedent("""
    import enum
    import cash


    class Color(enum.Enum):
        RED = 1
        BLUE = 2

        def bump(self, n):
            return self.extra + n


    class Planet(enum.Enum):
        EARTH = (5, 6)

        def __init__(self, mass, radius):
            self.mass = mass


    Color.RED.extra = 10
    CURRENT = Color.RED


    def helper(n):
        return Color.RED.bump(n)


    @cash.cache
    def through_helper(n):
        return helper(n)


    @cash.cache
    def through_global(n):
        return CURRENT.extra + n


    @cash.cache
    def through_argument(member, n):
        return member.extra + n


    @cash.cache
    def planet_mass(n):
        return Planet.EARTH.mass + n


    def show():
        return [
            [through_helper(1), through_global(1), through_argument(Color.RED, 1), planet_mass(1)]
            for _ in range(3)
        ]


    print(show())
    Color.RED.__dict__["extra"] = 20
    print(show())
    setattr(Color.RED, "extra", 30)
    Planet.EARTH.mass = 7
    print(show())
    del Color.RED.extra
    Color.RED.extra = 40
    print(show())
""")


def test_an_enum_member_attribute_change_matches_plain_python(tmp_path):
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    plain = run_python("main.py", cwd=tmp_path, env={"CASH_DISABLE": "1"}).stdout
    cached = run_python("main.py", cwd=tmp_path).stdout
    assert cached == plain, f"cash printed\n{cached}\nplain Python printed\n{plain}"
    # And a second process, from the entries the first stored.
    again = run_python("main.py", cwd=tmp_path).stdout
    assert again == plain
