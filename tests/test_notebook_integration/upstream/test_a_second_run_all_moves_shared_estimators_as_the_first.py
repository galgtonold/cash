"""A second Run All leaves the estimators an ensemble holds where the first did.

``voting_clf.set_params(svm_clf="drop")`` changes the ensemble in place, so
cash moves on the lineage of each variable that shares an object with it:
the estimators it was built from, the list of them. It moved each one on
once per partner that led to it, so a variable a cell below binds afresh
and that was still there from the last pass (``t =
voting_clf.named_estimators_.pop(...)``) made one more step: on the second
Run All the estimators came out with other lineages than the first gave
them, and the score cells below re-ran the cells above, the training loop
with them, printing its lines again. In E8 (homl ch07) 12 to 40 statements
above small cells, 40 s for a cell plain runs in 0.4 s.
"""

import pytest

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

MARK = (
    "import os\nLOG = os.path.abspath('calls.log')\n"
    "def mark(step):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (step + '\\n').encode())\n"
    "    os.close(fd)"
)


def _data(rows):
    return (
        "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\n"
        "from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import VotingClassifier\n"
        f"X = np.arange({rows * 2}.0).reshape({rows}, 2) % 7\n"
        "y = (X[:, 0] > 3).astype(int)"
    )


CELLS = [
    "import cash\n%cash_on",
    MARK,
    _data(40),
    "a = LogisticRegression()\nb = DecisionTreeClassifier(random_state=0)",
    "ests = [a, b]\nfor e in ests:\n    mark('fit')\n    print('Training the', type(e).__name__)\n    e.fit(X, y)",
    "named = [('a', a), ('b', b)]",
    "v = VotingClassifier(named)",
    "v.fit(X, y)",
    "v.set_params(b='drop')",
    "t = v.named_estimators_.pop('b')\nv.estimators_.remove(t)",
    "print('score', v.score(X, y))",
]


def _calls(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def test_a_second_run_all_reruns_nothing_above(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(len(CELLS))

    nb_runner.run_all()

    out = nb_runner.get_output(len(CELLS))
    assert "score" in first and first.strip() == out.strip(), (first, out)
    assert "Training" not in out, out
    # The loop cell itself runs on each pass, as plain Run All runs it.
    assert _calls(nb_runner) == ["fit"] * 4, _calls(nb_runner)


def test_an_edit_above_still_refits_the_estimators(nb_runner):
    """The control: the data edited, the score cell refits what it reads."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, _data(30))

    nb_runner.run_cell(len(CELLS))

    assert "score" in nb_runner.get_output(len(CELLS))
    assert _calls(nb_runner) == ["fit"] * 4, _calls(nb_runner)
