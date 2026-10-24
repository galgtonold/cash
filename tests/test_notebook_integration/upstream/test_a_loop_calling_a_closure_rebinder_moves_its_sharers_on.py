"""A loop whose body calls a function that rebinds its closure changes the
functions sharing the cell, as a statement calling it does.

``for f in [rebind]: f([0])`` ran as body statements, which leave what they
change to the loop, and the loop counted only the names its text changes:
``draw``, reading the rebound cell, kept the lineage ``make(BIG)`` gave it,
so the next cell reading it rebuilt ``draw`` and ran the loop again inside
that cell, printing what the loop prints, in a plain Run All.
"""

MAKE = (
    "def make(rows):\n    def draw(k):\n        return len(rows)\n"
    "    def rebind(r):\n        nonlocal rows\n        print('rebind ran')\n        rows = r\n    return draw, rebind\n"
    "BIG = [{'id': i} for i in range(20000)]"
)


def test_the_reader_after_it_is_not_rebuilt_in_a_run_all(nb_runner):
    nb_runner.create_notebook(
        [MAKE, "draw, rebind = make(BIG)", "for f in [rebind]:\n    f([0])", "y = draw(1)\nprint(y)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "rebind ran" in nb_runner.get_output(3), nb_runner.get_output(3)
    assert nb_runner.get_output(4).strip() == "1", f"the reader ran the loop again: {nb_runner.get_output(4)!r}"
