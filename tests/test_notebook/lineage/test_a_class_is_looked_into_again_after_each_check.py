"""What a walk learns about a class lasts for one upstream check only.

Inside ``callee_reach.one_walk`` -- the stretch of a check where no code of
the notebook runs -- each class is looked into once. A cell may give a
library class a method of its own between two checks
(``pd.DataFrame.report = report``), and a statement reading a frame then
reaches that method and what it reads: the next walk must see it.
"""

from __future__ import annotations

import fractions

import pytest

from cash.notebook import callee_reach


@pytest.fixture
def patched_fraction():
    """``fractions.Fraction`` given a notebook method by the test, removed after."""
    yield fractions.Fraction
    if "where_mode" in vars(fractions.Fraction):
        del fractions.Fraction.where_mode


def test_a_method_a_cell_gives_a_library_class_is_seen_by_the_next_walk(patched_fraction):
    ns: dict = {"__name__": "__main__", "fr": fractions.Fraction(1, 3)}
    code = "m = fr.where_mode()"
    with callee_reach.one_walk():
        assert callee_reach.reached_user_code(code, ns).functions == ()

    exec("import os\ndef where_mode(self):\n    return os.environ.get('MODE')", ns)
    patched_fraction.where_mode = ns["where_mode"]

    with callee_reach.one_walk():
        assert [f.__name__ for f in callee_reach.reached_user_code(code, ns).functions] == ["where_mode"]
    # And outside any walk, as the runtime asks.
    assert [f.__name__ for f in callee_reach.reached_user_code(code, ns).functions] == ["where_mode"]


def test_a_cells_class_is_followed_inside_a_walk():
    """Control: a class a cell defines leads to its methods, inside a walk
    as outside, for every statement that reads an instance."""
    ns: dict = {"__name__": "__main__"}
    exec("class Model:\n    def run(self):\n        return 1\nm = Model()", ns)
    with callee_reach.one_walk():
        for _ in range(3):
            assert [f.__name__ for f in callee_reach.reached_user_code("y = m.run()", ns).functions] == ["run"]


def test_a_class_met_again_inside_its_own_walk_is_still_followed():
    """A method reading an instance of its own class: telling whether the
    class leads anywhere meets the class again, which must not start over."""
    ns: dict = {"__name__": "__main__"}
    exec(
        "import os\nclass Node:\n    def peer_mode(self):\n        return SELF.mode()\n"
        "    def mode(self):\n        return os.environ.get('MODE')\nSELF = Node()\nn = Node()",
        ns,
    )
    with callee_reach.one_walk():
        names = {f.__name__ for f in callee_reach.reached_user_code("y = n.peer_mode()", ns).functions}
    assert names == {"peer_mode", "mode"}


def test_a_class_leading_nowhere_does_not_hide_another_that_does():
    """Two classes in one walk: the library one adds nothing, the cell's one
    still adds its method."""
    ns: dict = {"__name__": "__main__", "fr": fractions.Fraction(1, 2)}
    exec("class Box:\n    def get(self):\n        return 1\nb = Box()", ns)
    with callee_reach.one_walk():
        assert callee_reach.reached_user_code("x = fr + 1", ns) == callee_reach._EMPTY
        assert [f.__name__ for f in callee_reach.reached_user_code("y = fr + b.get()", ns).functions] == ["get"]
