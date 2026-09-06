"""CAS-96: a trailing ``;`` suppresses the repr, on first run and cached re-run.

``ast.unparse`` drops the trailing semicolon, so cash used to display (and cache)
the last expression's repr regardless — a phantom repr on cached re-runs. The
splitter now re-attaches the ``;`` so the suppression rides through both the
cache key and the execution path.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]


def test_semicolon_suppresses_repr_first_and_cached(nb_runner):
    nb_runner.create_notebook([
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
        "df.tail(2)",     # no semicolon -> displays
        "df.head(1);",    # semicolon -> suppressed
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "a" in nb_runner.get_output(2) and "3" in nb_runner.get_output(2)
    assert nb_runner.get_output(3).strip() == "", nb_runner.get_output(3)

    # Cached re-run: suppression must persist (no phantom repr).
    nb_runner.run_all()
    assert "a" in nb_runner.get_output(2) and "3" in nb_runner.get_output(2)
    assert nb_runner.get_output(3).strip() == "", nb_runner.get_output(3)


def test_non_suppressed_expression_still_displays_on_cached_run(nb_runner):
    nb_runner.create_notebook([
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
        "df.head(2)",     # no semicolon -> must still display on both runs
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "a" in nb_runner.get_output(2) and "1" in nb_runner.get_output(2)

    nb_runner.run_all()
    assert "a" in nb_runner.get_output(2) and "1" in nb_runner.get_output(2), (
        f"cached re-run lost the display: {nb_runner.get_output(2)!r}"
    )


def test_semicolon_with_trailing_comment_suppresses(nb_runner):
    nb_runner.create_notebook([
        "x = 41",
        "x + 1;  # answer, suppressed",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert nb_runner.get_output(2).strip() == "", nb_runner.get_output(2)
    nb_runner.run_all()
    assert nb_runner.get_output(2).strip() == "", nb_runner.get_output(2)


def test_semicolon_suppresses_after_a_non_ascii_character(nb_runner):
    """``end_col_offset`` is a UTF-8 byte offset. Reading it as a character
    index slid the slice past the ``;`` whenever anything non-ASCII sat
    earlier on the same line — so a perfectly ordinary filter on a city name
    echoed a repr the user had suppressed. The unsuppressed arm is what proves
    the suppressed arm isn't passing because output is broken generally."""
    nb_runner.create_notebook([
        "import pandas as pd\n"
        "df = pd.DataFrame({'city': ['Zürich', 'Genève'], 'n': [1, 2]})",
        "df[df.city == 'Zürich']",     # no semicolon -> displays
        "df[df.city == 'Zürich'];",    # semicolon -> suppressed
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "Zürich" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert nb_runner.get_output(3).strip() == "", nb_runner.get_output(3)

    nb_runner.run_all()
    assert "Zürich" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert nb_runner.get_output(3).strip() == "", nb_runner.get_output(3)
