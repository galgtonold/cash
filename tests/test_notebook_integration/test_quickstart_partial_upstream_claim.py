"""Guard for the quickstart's headline claim and the badge it ships.

``docs/getting-started/quickstart.md`` (§ "Change one thing upstream — run only
the cell you care about") tells the reader that after editing ``THRESHOLD`` they
may run *only* the downstream cell, and that cash will repair a **part** of the
cell in between: the cheap THRESHOLD-dependent statement re-runs while the
expensive THRESHOLD-independent one does not.

``scripts/badge_fixtures.py::quickstart_partial_upstream`` renders that cell's
badge for the page, and its comment claims the row set was captured from a real
kernel — two upstream rows, with ``features = build_features()`` absent. Both
halves are asserted here so the page cannot drift away from the engine.

**Counted, not timed.** Wall-clock cannot distinguish "recomputed" from
"restored but slow", so each helper appends a line to a log file the test reads.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _counts(log):
    """Return {marker: times executed} from the helper log."""
    if not log.exists():
        return {}
    out = {}
    for line in log.read_text().splitlines():
        out[line] = out.get(line, 0) + 1
    return out


@pytest.fixture
def threshold_notebook(nb_runner, tmp_path):
    """The quickstart's three-cell shape, plus a text badge and a call log."""
    log = tmp_path / "calls.log"
    setup = f"""
import time, pathlib
LOG = pathlib.Path(r"{log}")

def build_features():
    with LOG.open("a") as fh:
        fh.write("build\\n")
    time.sleep(0.4)
    return list(range(20))

def score(feats, thr):
    with LOG.open("a") as fh:
        fh.write("score\\n")
    return [f for f in feats if f > thr]
"""
    nb_runner.create_notebook([
        "%cash_badge print",                      # 1 — text badge, so it is readable
        setup,                                    # 2 — helpers
        "THRESHOLD = 10",                         # 3 — the parameter
        "features = build_features()\n"           # 4 — expensive, THRESHOLD-independent
        "flagged  = score(features, THRESHOLD)",  #     cheap, THRESHOLD-dependent
        "print('N', len(flagged))",               # 5 — downstream
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "N 9" in nb_runner.get_output(5), "baseline Run All is wrong"
    # Non-vacuity: the helpers must actually have run, or every delta below is
    # trivially satisfied.
    assert _counts(log) == {"build": 1, "score": 1}, _counts(log)
    return nb_runner, log


def test_only_the_threshold_dependent_statement_reruns(threshold_notebook):
    """Edit THRESHOLD, run ONLY cell 5: score() re-runs, build_features() does not."""
    nb_runner, log = threshold_notebook
    before = _counts(log)

    nb_runner.set_cell_source(3, "THRESHOLD = 15")
    nb_runner.run_cell(5)  # NOT cell 4, NOT Run All

    after = _counts(log)
    assert "N 4" in nb_runner.get_output(5), "downstream did not see the new THRESHOLD"
    assert after.get("build", 0) == before["build"], (
        "the expensive THRESHOLD-independent statement re-ran; the quickstart "
        f"claims it does not (before={before}, after={after})"
    )
    assert after.get("score", 0) == before["score"] + 1, (
        f"the THRESHOLD-dependent statement did not re-run exactly once "
        f"(before={before}, after={after})"
    )


def test_badge_upstream_rows_match_the_docs_fixture(threshold_notebook):
    """The badge lists the two repaired statements and NOT the expensive one.

    Mirrors ``quickstart_partial_upstream`` in ``scripts/badge_fixtures.py``,
    which docs/getting-started/quickstart.md embeds.
    """
    nb_runner, _ = threshold_notebook
    nb_runner.set_cell_source(3, "THRESHOLD = 15")
    nb_runner.run_cell(5)
    badge = nb_runner.get_output(5)

    assert "Upstream:" in badge, badge
    assert "THRESHOLD = 15" in badge, badge
    assert "flagged = score(features, THRESHOLD)" in badge, badge
    assert "build_features" not in badge, (
        "the expensive statement appeared on the badge; the docs fixture omits "
        f"it deliberately and would now be wrong:\n{badge}"
    )
