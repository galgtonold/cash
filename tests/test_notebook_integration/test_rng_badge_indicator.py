"""The badge marks statements with an RNG effect (Stage 1 of the randomness UX).

A seed/draw statement carries a text pill (seed / random) and a "Random" drawer
line; an *unseeded* draw's pill uses the warn-red family, flags that its cached
value is a frozen replay, and bumps the badge's header warning count. Behaviour
is unchanged — this is advisory only.
"""
import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"


def _badge_html(nb_runner, cell_num):
    """The rendered badge HTML for a cell (the display_data text/html output)."""
    cell = nb_runner.nb.cells[cell_num - 1]
    for o in cell.get("outputs", []):
        data = o.get("data") if isinstance(o, dict) else None
        if isinstance(data, dict) and data.get("text/html"):
            html = data["text/html"]
            if "c3-" in html:  # the interactive badge, not some other display
                return html
    return ""


@pytest.mark.timeout(180)
def test_unseeded_draw_flagged(nb_runner):
    nb_runner.create_notebook([C_ON, "import numpy as np\nu = np.random.rand(3)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 2)
    # Target the span USAGE, not the class name (which also appears in the <style>).
    assert 'class="c3-rng-pill c3-rng-warn"' in html, "unseeded draw not styled as a warning pill"
    assert ">unseeded<" in html, "unseeded draw pill should read 'unseeded'"
    assert "frozen replay" in html, "unseeded draw drawer missing the frozen-replay note"


@pytest.mark.timeout(180)
def test_seeded_draw_marked_but_not_warned(nb_runner):
    nb_runner.create_notebook([C_ON, "import numpy as np\nnp.random.seed(0)\ns = np.random.rand(3)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 2)
    assert ">random<" in html, "seeded draw pill should read 'random'"
    assert "reproducible" in html, "seeded draw drawer should say reproducible"
    assert 'c3-rng-warn"' not in html, "a seeded draw must not use the warn pill"


@pytest.mark.timeout(180)
def test_reexecuted_seed_explains_why(nb_runner):
    """A seed re-run only to re-establish the stream is labeled on the badge (Stage 2)."""
    data = "import numpy as np\nnp.random.seed(42)\nMULT = 100.0\narr = np.random.rand(3) * MULT"
    nb_runner.create_notebook([C_ON, data, "total = float(arr.sum())\nprint('T', total)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, data.replace("MULT = 100.0", "MULT = 200.0"))
    nb_runner.run_cell(3)  # downstream-only re-run forces the seed to re-run
    html = _badge_html(nb_runner, 3)
    assert "re-run to restore the random stream" in html, (
        "the re-executed seed should explain why it re-ran"
    )


_SK_SETUP = (
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.datasets import make_classification\n"
    "X, y = make_classification(n_samples=300, random_state=0)"
)


@pytest.mark.timeout(240)
def test_inline_unseeded_fit_flagged(nb_runner):
    """An inline/anonymous unseeded estimator fit is flagged on the badge (Stage 3)."""
    nb_runner.create_notebook([
        C_ON, _SK_SETUP,
        "clf = RandomForestClassifier(n_estimators=40).fit(X, y)\nprint('S', round(clf.score(X, y), 3))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 3)
    assert ">unseeded<" in html, "inline unseeded fit should show the unseeded pill"
    assert "frozen replay" in html, "inline unseeded fit drawer missing the frozen-replay note"


@pytest.mark.timeout(240)
def test_inline_seeded_fit_not_flagged(nb_runner):
    """A seeded inline fit must not be flagged as unseeded (Stage 3)."""
    nb_runner.create_notebook([
        C_ON, _SK_SETUP,
        "clf = RandomForestClassifier(n_estimators=40, random_state=1).fit(X, y)\nprint('S', round(clf.score(X, y), 3))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 3)
    assert 'c3-rng-warn"' not in html, "a seeded inline fit must not use the warn pill"


_HELPER_UNSEEDED = (
    "import numpy as np\n"
    "def make_data():\n"
    "    return np.random.rand(3)  # unseeded draw, hidden from the AST"
)


@pytest.mark.timeout(180)
def test_helper_internal_unseeded_draw_flagged(nb_runner):
    """A draw hidden inside a called function is caught by the runtime observer.

    ``x = make_data()`` spells no ``np.random`` and binds a plain array (nothing
    to introspect), so both static analysis and object-introspection are blind.
    The before/after global-RNG diff sees the stream advance and flags the frozen
    replay.
    """
    nb_runner.create_notebook([C_ON, _HELPER_UNSEEDED, "x = make_data()"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 3)
    assert ">unseeded<" in html, "helper-internal unseeded draw should show the unseeded pill"
    assert "frozen replay" in html, "helper-internal draw drawer missing the frozen-replay note"


@pytest.mark.timeout(180)
def test_bare_uncached_fit_still_no_pill(nb_runner):
    """A bare ``model.fit()`` is skip-cache (recomputes fresh) so nothing is frozen.

    The observer sees the global stream advance, but the ``skip_cache`` gate keeps
    the pill off: there is no frozen replay to warn about. This is the F2 case —
    absence of the pill is correct, not a miss.
    """
    setup = (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.datasets import make_classification\n"
        "X, y = make_classification(n_samples=200, random_state=0)\n"
        "clf = RandomForestClassifier(n_estimators=20)"
    )
    nb_runner.create_notebook([C_ON, setup, "clf.fit(X, y)"])  # bare, no @cash:cache-fit
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 3)
    assert 'class="c3-rng-pill' not in html, "an uncached bare fit must carry no random pill"


@pytest.mark.timeout(180)
def test_non_random_statement_has_no_chip(nb_runner):
    nb_runner.create_notebook([C_ON, "a = 1 + 1\nb = a * 2"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 2)
    assert html, "expected a badge for the cell"
    assert 'class="c3-rng-pill' not in html, "non-random statement should carry no random pill"
