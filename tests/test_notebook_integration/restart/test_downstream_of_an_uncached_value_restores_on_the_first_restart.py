"""A statement reading a value cash never stores restores on the FIRST restart.

``A = a.fit_transform(data)`` is not cached (it fits an estimator), so every
kernel runs it again. ``s = heavy(A, B, C)`` reads what three such fits return
and is cached. The key the first kernel recorded for ``s`` must be the key every
later kernel computes: otherwise the first Restart & Run All after a session
runs ``heavy`` again, and only a later restart hits.

A fit writes two names, the result and the fitted estimator, and the lineage of
the result depended on which of the two was recorded first -- set order, which
each kernel's string hashing picks anew. Three fits make it very unlikely that
three kernels happen to agree by chance.
"""

import pytest

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import time\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler",
    "data = np.random.default_rng(0).normal(size=(500, 4))",
    "a = StandardScaler()\nb = StandardScaler()\nc = StandardScaler()",
    "A = a.fit_transform(data)\nB = b.fit_transform(data)\nC = c.fit_transform(data)",
    "def heavy(*ms):\n    time.sleep(0.3)\n    return round(sum(float((m ** 2).sum()) for m in ms), 6)",
    "s = heavy(A, B, C)\nprint('S', s)",
]
FIT, HEAVY = 5, 7
EXPECTED = "S 6000.0"


def _heavy_row(out):
    rows = [r.strip() for r in out.splitlines() if ": s = heavy(A, B, C)" in r]
    assert len(rows) == 1, out
    return rows[0]


@pytest.mark.parametrize("annotate", [False, True], ids=["plain-fit", "cache-fit"])
def test_the_first_restart_restores_the_downstream_statement(nb_runner, annotate):
    cells = list(CELLS)
    if annotate:
        cells[FIT - 1] = "\n".join("# @cash:cache-fit\n" + line for line in cells[FIT - 1].splitlines())
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(HEAVY)
    assert EXPECTED in first, first
    assert _heavy_row(first).startswith("EXECUTED:"), first

    for kernel in (2, 3):
        nb_runner.restart()
        nb_runner.run_all()
        fit, out = nb_runner.get_output(FIT), nb_runner.get_output(HEAVY)
        assert EXPECTED in out, f"kernel {kernel}:\n{out}"
        assert _heavy_row(out).startswith("CACHED:"), f"kernel {kernel} ran heavy again:\n{fit}\n{out}"


# A cleaning cell that fits encoders into the frame it rebuilds: each
# ``clean[...] = enc.fit_transform(...)`` writes ``clean`` and the encoder, so the
# self-rebinding ``clean = clean.sort_values(...)`` below it was keyed on
# whichever the kernel recorded first. After an edit to that line, the upstream
# replay run from the cell below keyed it one way and the next kernel the other.
CLEAN = (
    "clean = raw.dropna()\n"
    "clean['head'] = le_head.fit_transform(clean['text'].str[:3])\n"
    "clean['tail'] = le_tail.fit_transform(clean['text'].str[-3:])\n"
    "clean = clean.sort_values('id')"
)
TEXT_CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import time\nimport numpy as np\nimport pandas as pd\n"
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "from sklearn.preprocessing import LabelEncoder",
    "rng = np.random.default_rng(0)\n"
    "words = np.array(['alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta', 'theta'])\n"
    "raw = pd.DataFrame({'id': rng.permutation(3000), 'text': [' '.join(rng.choice(words, 6)) for _ in range(3000)]})",
    "le_head = LabelEncoder()\nle_tail = LabelEncoder()",
    CLEAN,
    "tfidf = TfidfVectorizer()\nX = tfidf.fit_transform(clean['text'])",
    "def heavy(m):\n    time.sleep(0.3)\n    return round(float(m.sum()), 4)",
    "s = heavy(X)\nprint('S', s)",
]


def test_an_edited_self_rebinding_line_keeps_its_key_across_restarts(nb_runner):
    nb_runner.create_notebook(TEXT_CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(5, CLEAN + ".drop_duplicates('text')")
    # Run only the last cell: the edited cleaning line re-runs as its upstream.
    nb_runner.run_cell(8)
    edited = nb_runner.get_output(8)
    assert "EXECUTED: s = heavy(X)" in edited, edited

    for kernel in (2, 3):
        nb_runner.restart()
        nb_runner.run_all()
        out = nb_runner.get_output(8)
        rows = [r.strip() for r in out.splitlines() if ": s = heavy(X)" in r]
        assert len(rows) == 1 and rows[0].startswith("CACHED:"), f"kernel {kernel} ran heavy again:\n{out}"
