"""An expensive call handed to another call is cached on its own.

Round 30, r30s3: ``Wy = weights_from_scores(fit_predict(feats, yr), px.columns,
rebal_every=5).shift(1).fillna(0.0)`` re-ran all nine fits (19 s), although the
same ``fit_predict(feats, yr)`` restored when written on its own line. "Written
on its own line they restore; nested as an argument they don't."

Only the outer call was taken, and its subtree was then left alone. Wrapping
a call replaces its callee expression only, so an argument runs whether the
outer call hits or not -- taking the inner one as well costs nothing and is
the only way its work is ever reused.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = (
    "import cash\n%cash_on\n%cash_badge print\nimport os, time\n"
    "def mark(name):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, name.encode())\n    os.close(fd)\n"
    "def fit(xs, k):\n    mark('f')\n    time.sleep(0.4)\n    return [x * k for x in xs]\n"
    "def weights(scores, cap):\n    mark('w')\n    return sum(scores) + cap\n"
    "data = list(range(200))\nK = 3\n"
)
NESTED = "out = weights(fit(data, K), 10)\nprint('OUT', out)"


def _runs(runner, name) -> int:
    log = Path(runner.work_dir) / "runs.log"
    return log.read_text().count(name) if log.exists() else 0


def test_the_nested_call_is_restored(nb_runner):
    nb_runner.create_notebook([SETUP, NESTED, "print('TAIL')"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT 59710" in nb_runner.get_output(2), nb_runner.get_raw_output(2)
    assert _runs(nb_runner, "f") == 1

    # An edit to the statement around it: the fit itself has not changed.
    nb_runner.set_cell_source(2, "out = weights(fit(data, K), 11)\nprint('OUT', out)")
    nb_runner.run_cell(2)
    assert "OUT 59711" in nb_runner.get_output(2), nb_runner.get_raw_output(2)
    assert _runs(nb_runner, "f") == 1, (
        "the nested fit ran again although its arguments did not change:\n" + nb_runner.get_raw_output(2)
    )


def test_a_changed_argument_still_re_runs_the_nested_call(nb_runner):
    """Control: the inner call's own inputs decide, as on its own line."""
    nb_runner.create_notebook([SETUP, NESTED, "print('TAIL')"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner, "f") == 1

    nb_runner.set_cell_source(1, SETUP.replace("K = 3", "K = 4"))
    nb_runner.run_all()
    assert "OUT 79610" in nb_runner.get_output(2), nb_runner.get_raw_output(2)
    assert _runs(nb_runner, "f") == 2
