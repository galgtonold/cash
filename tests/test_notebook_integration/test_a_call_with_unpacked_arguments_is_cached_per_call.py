"""A call whose arguments are unpacked (`f(g, **TUNED.get(k, {}))`) is cached per call.

Round 25's r25s5 fitted one model per store and department, with per-department
settings::

    models = {key: fit_series(g, **TUNED.get(key[1], {})) for key, g in series.items()}

Growing the slice from 6 groups to 7 re-fitted all 7, with no sub-call line in
the badge; spelled ``fit_series(g, max_depth=3)`` it served 6 of 7. A call with
``*``/``**`` unpacking was refused outright, since its argument positions are
not known before it runs (CAS-243: ``compute(*pair())`` had keyed only the
first of two values). Keyed on what it receives, the call hashes what did
arrive -- every positional value, and every keyword with its name.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

SETUP = (
    "import os, time\n"
    "def fit(values, depth=1, rate=1.0):\n"
    "    fd = os.open('fits.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'fit|')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.02)\n"
    "    return round(sum(values) * depth * rate, 6)\n"
    "TUNED = {'a': {'depth': 2}, 'b': {'depth': 3, 'rate': 0.5}}\n"
    "groups = {(s, d): [float(s + i) for i in range(5)] for s in range(4) for d in 'abc'}"
)
FIT = (
    "few = dict(list(groups.items())[:{n}])\n"
    "models = {{key: fit(g, **TUNED.get(key[1], {{}})) for key, g in few.items()}}\n"
    "print('MODELS', sorted(models.items()))"
)


def _fits(runner) -> int:
    log = Path(runner.work_dir) / "fits.log"
    return log.read_text().count("fit|") if log.exists() else 0


def _expected(n):
    tuned = {"a": {"depth": 2}, "b": {"depth": 3, "rate": 0.5}}
    groups = {(s, d): [float(s + i) for i in range(5)] for s in range(4) for d in "abc"}
    out = {}
    for key, g in list(groups.items())[:n]:
        kw = tuned.get(key[1], {})
        out[key] = round(sum(g) * kw.get("depth", 1) * kw.get("rate", 1.0), 6)
    return f"MODELS {sorted(out.items())}"


def test_growing_the_slice_fits_only_the_new_group(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, FIT.format(n=6)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _expected(6) in nb_runner.get_output(3), nb_runner.get_output(3)
    assert _fits(nb_runner) == 6

    nb_runner.set_cell_source(3, FIT.format(n=7))
    nb_runner.run_cell(3)
    assert _expected(7) in nb_runner.get_output(3), nb_runner.get_output(3)
    assert _fits(nb_runner) == 7, "every group was re-fitted"


def test_a_changed_setting_refits_only_its_groups(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, FIT.format(n=12)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _fits(nb_runner) == 12

    nb_runner.set_cell_source(2, SETUP.replace("'rate': 0.5", "'rate': 0.25"))
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    tuned = {"a": {"depth": 2}, "b": {"depth": 3, "rate": 0.25}}
    groups = {(s, d): [float(s + i) for i in range(5)] for s in range(4) for d in "abc"}
    expected = sorted(
        (k, round(sum(g) * tuned.get(k[1], {}).get("depth", 1) * tuned.get(k[1], {}).get("rate", 1.0), 6))
        for k, g in groups.items()
    )
    assert f"MODELS {expected}" in out, out
    assert _fits(nb_runner) == 16, "only the four 'b' groups have a new setting"


def test_star_arguments_are_keyed_on_every_value(nb_runner):
    cells = [
        "import cash\n%cash_on",
        SETUP,
        "pairs = [(1.0, 2.0), (1.0, 3.0), (1.0, 2.0)]\nout = [fit(list(p), *p[1:]) for p in pairs]\nprint('OUT', out)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT [6.0, 12.0, 6.0]" in nb_runner.get_output(3), nb_runner.get_output(3)
    assert _fits(nb_runner) == 2
