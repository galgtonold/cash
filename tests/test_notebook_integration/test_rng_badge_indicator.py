"""The badge marks statements with an RNG effect (Stage 1 of the randomness UX).

A seed/draw statement carries a 🎲 chip and a "Random" drawer line; an *unseeded*
draw additionally flags that its cached value is a frozen replay and bumps the
badge's header warning count. Behaviour is unchanged — this is advisory only.
"""
import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
DIE = chr(0x1F3B2)


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
    assert DIE in html, "unseeded draw missing the random chip"
    assert "c3-rng-warn" in html, "unseeded draw not styled as a warning"
    assert "frozen replay" in html, "unseeded draw drawer missing the frozen-replay note"


@pytest.mark.timeout(180)
def test_seeded_draw_marked_but_not_warned(nb_runner):
    nb_runner.create_notebook([C_ON, "import numpy as np\nnp.random.seed(0)\ns = np.random.rand(3)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 2)
    assert DIE in html, "seeded draw missing the random chip"
    assert "reproducible" in html, "seeded draw drawer should say reproducible"
    assert "c3-rng-warn" not in html, "a seeded draw must not be flagged as unseeded"


@pytest.mark.timeout(180)
def test_non_random_statement_has_no_chip(nb_runner):
    nb_runner.create_notebook([C_ON, "a = 1 + 1\nb = a * 2"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    html = _badge_html(nb_runner, 2)
    assert html, "expected a badge for the cell"
    assert DIE not in html and "c3-rng" not in html, "non-random statement should carry no random chip"
