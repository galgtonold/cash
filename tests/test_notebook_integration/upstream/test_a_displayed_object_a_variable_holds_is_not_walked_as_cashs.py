"""What the output history holds, and a variable too, is not walked as cash's.

``voting_clf.fit(X_train, y_train)`` shows the ensemble, so the output
history holds it with its fitted forests. The share check of each
statement whose output another variable shares counts what cash holds,
the output history among it, and walked those forests a tree at a time, in
each of its rounds: past its 0.1 s budget for
``voting_clf.estimators_.remove(svm_clf_trained)``, and the search of the
whole heap that follows took a second more, on each pass of E8 (homl ch07).
An object a variable is bound to is the user's on its face: the walk stops
there.
"""

import contextlib
import io

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

FOREST = (
    "class Forest:\n"
    "    def __init__(self, k):\n"
    "        self.trees = [{'n': [i, i + 1], 'c': [i]} for i in range(k)]\n"
    "    def fit(self):\n"
    "        return self"
)
K = 20_000


def _cells(shown: str) -> list[str]:
    return [
        "import cash\n%cash_on",
        FOREST,
        f"forest = Forest({K})",
        shown,
        "class Box:\n    pass\nholder = {'b': Box()}",
        "item = holder['b']",
        "print('same', item is holder['b'], len(forest.trees))",
    ]


#: Counts, in the kernel, the containers the walk of what cash holds reads.
#: Self-contained (nothing of it in the notebook's globals) and undone by
#: `UNDO`: a kernel reused by the next test keeps its modules.
COUNT = (
    "def _cash_test_count(so=__import__('cash.notebook.shared_objects', fromlist=['_'])):\n"
    "    real = so.__dict__.setdefault('_real_walk_held', so._walk_held)\n"
    "    so._walked = []\n"
    "    def counting(*a, **k):\n"
    "        out = real(*a, **k)\n"
    "        so._walked.append(len(out[2]))\n"
    "        return out\n"
    "    so._walk_held = counting\n"
    "_cash_test_count()\n"
    "del _cash_test_count"
)
UNDO = (
    "(lambda so: setattr(so, '_walk_held', so.__dict__.pop('_real_walk_held', so._walk_held)))"
    "(__import__('cash.notebook.shared_objects', fromlist=['_']))"
)


def _plain(cells: list[str]) -> str:
    """What a plain Run All of *cells* prints (no cash, no IPython)."""
    ns: dict = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for code in cells[1:]:
            exec(code, ns)
    return out.getvalue().strip()


def _walked_by_the_share_check(nb_runner, cells: list[str]) -> int:
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for cell in range(1, 6):
        nb_runner.run_cell(cell)
    nb_runner.peek(f"exec({COUNT!r})")
    try:
        nb_runner.run_cell(6)
        walked = int(nb_runner.peek("sum(__import__('cash.notebook.shared_objects', fromlist=['_'])._walked)"))
    finally:
        nb_runner.peek(UNDO)
    nb_runner.run_cell(7)
    assert nb_runner.get_output(7).strip() == _plain(cells), nb_runner.get_output(7)
    assert "SHARE-UNCHECKED" not in nb_runner.get_raw_output(6), nb_runner.get_raw_output(6)
    return walked


@pytest.mark.fresh_kernel
def test_a_shown_object_a_variable_holds_is_not_walked(nb_runner):
    walked = _walked_by_the_share_check(nb_runner, _cells("forest.fit()"))
    assert walked < 1_000, f"walked {walked} containers of what cash holds"


@pytest.mark.fresh_kernel
def test_one_no_variable_holds_still_is(nb_runner):
    """The control: shown and bound to no variable, it is cash's own, and
    the walk reads into it."""
    walked = _walked_by_the_share_check(nb_runner, _cells(f"Forest({K})"))
    assert walked >= K, f"walked {walked} containers of what cash holds"
