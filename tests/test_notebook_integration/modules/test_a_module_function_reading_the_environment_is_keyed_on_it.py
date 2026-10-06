"""A cell calling a local module's function that reads an environment variable follows the variable.

``m = mylib.mode()``, with ``mode`` reading ``os.environ["MODE"]``, and a
cell above setting ``MODE``: editing that cell and running the notebook again
served the old mode, with no warning. The same function written in a cell,
or the read written in the statement, was right.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = "import os, time\ndef mode():\n    time.sleep(0.3)\n    return os.environ.get('CASH_IT_MODE', 'none')\n"


@pytest.mark.parametrize(
    "setter",
    ["import os\nos.environ['CASH_IT_MODE'] = '{v}'", "%env CASH_IT_MODE={v}"],
    ids=["os_environ", "env_magic"],
)
def test_editing_the_cell_that_sets_it_and_running_all(nb_runner, tmp_path, setter):
    (tmp_path / "envmodlib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(
        ["%cash_badge print\nimport envmodlib", setter.format(v="train"), "m = envmodlib.mode()\nprint('M', m)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "M train" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, setter.format(v="test"))
    nb_runner.run_all()
    assert nb_runner.peek("m") == "'test'", nb_runner.get_raw_output(3)

    # Back to the first value: its entry is still there, and the call is restored.
    nb_runner.set_cell_source(2, setter.format(v="train"))
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(3)
    assert nb_runner.peek("m") == "'train'"
    assert "CACHED: m = envmodlib.mode()" in raw, raw
