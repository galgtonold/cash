"""A cached call inside a comprehension is keyed by each element's value.

Round 22, found by the tester-session tests (r22s2's notebook): a grid
search per model, ``models = {name: fit_full(s, X, y) for name, s in
searches.items()}``, handed the SVM the logistic regression. The call key
resolved the comprehension's variable ``s`` by NAME -- finding nothing, or an
unrelated global ``s`` from another cell -- so every element had one key, and
the second call was served the first's result. With a global of the same name
around it was wrong on the very first run.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

SETUP = ("import time\n"
         "def slow(v):\n"
         "    open('calls.log', 'a').write(f'slow {v}\\n')\n"
         "    time.sleep(0.05)\n"
         "    return v * 10\n"
         "d = {'a': 1, 'b': 2}")


def _calls(runner) -> list[str]:
    log = Path(runner.work_dir) / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize("cell", [
    "v = 'a global named like the variable'\nout = {k: slow(v) for k, v in d.items()}\nprint(out)",
    "out = {k: slow(v) for k, v in d.items()}\nprint(out)\nv = 'bound after the first run'",
    "out = [slow(v) for v in d.values()]\nprint(dict(zip(d, out)))",
    "out = dict(map(lambda kv: (kv[0], slow(kv[1])), d.items()))\nprint(out)",
], ids=["global_same_name", "global_bound_later", "list_comp", "lambda"])
def test_each_element_gets_its_own_result(nb_runner, cell):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "'a': 10, 'b': 20" in nb_runner.get_output(3), nb_runner.get_output(3)

    # One element changes: its call recomputes, the other still comes back.
    nb_runner.set_cell_source(2, SETUP.replace("{'a': 1, 'b': 2}", "{'a': 1, 'b': 5}"))
    before = len(_calls(nb_runner))
    nb_runner.run_all()
    assert "'a': 10, 'b': 50" in nb_runner.get_output(3), nb_runner.get_output(3)
    assert "slow 5" in _calls(nb_runner)[before:]


def test_an_argument_built_from_the_element_is_keyed_by_all_of_it(nb_runner):
    """``fit_score(make_features(cleaned[mid], W)) for mid in ids`` (r23s3):
    the argument is computed from the element, and only its value tells the
    elements apart. That value was hashed from a sample -- a frame's shape,
    dtypes and first five rows -- and rolling-window features all begin with
    the same empty rows: two elements, one key, the second served the first's
    result on a first run. It takes a global named like the variable for the
    call to be cached at all (r23s3 had one from an earlier ``for`` loop)."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "import time\nimport pandas as pd\n"
        "def frame(k):\n"
        "    return pd.DataFrame({'v': [0.0] * 5 + [float(k)] * 5})\n"
        "def slow_sum(df):\n"
        "    time.sleep(0.05)\n"
        "    return float(df['v'].sum())\n"
        "keys = [1, 2]\n"
        "for k in keys:\n"
        "    pass",
        "out = {k: slow_sum(frame(k)) for k in keys}\nprint(out)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "{1: 5.0, 2: 10.0}" in nb_runner.get_output(3), nb_runner.get_output(3)


def test_a_callee_that_is_the_element_is_not_intercepted(nb_runner):
    """`m.predict(...)` over `models.items()` is a different callable per
    element, which no key can see: it must simply run."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "class M:\n"
        "    def __init__(self, k): self.k = k\n"
        "    def predict(self, x):\n"
        "        import time; time.sleep(0.05)\n"
        "        return x * self.k\n"
        "models = {'a': M(2), 'b': M(3)}\nx = 10",
        "preds = {name: m.predict(x) for name, m in models.items()}\nprint(preds)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert "{'a': 20, 'b': 30}" in nb_runner.get_output(3), nb_runner.get_output(3)
