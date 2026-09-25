"""A submodule bound by ``from pkg import helpers`` keeps its key between runs.

``import pkg`` tracked the package; ``from pkg import helpers`` tracked only
``pkg``, not ``pkg.helpers``. The first run keyed a statement reading
``helpers.measure`` on the module's whole lineage; by the second run the
submodule had been picked up from the namespace, its key was narrowed to what
the statement reads, and the second run missed and recomputed. Only the third
run hit.

Each call is counted with ``os.open``/``os.write``, which a cache hit does not
replay; the sleep puts the statement above the store floor.
"""

import pytest

pytestmark = [pytest.mark.integration]

HELPERS = """import os
import time

def measure(text):
    time.sleep(0.2)
    fd = os.open("calls.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.write(fd, b"call\\n")
    os.close(fd)
    return len(text)
"""


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        ("from pkg import helpers", 'n = helpers.measure("abc")'),
        ("from pkg import helpers as h", 'n = h.measure("abc")'),
        ("import pkg.helpers", 'n = pkg.helpers.measure("abc")'),
        ("import helpers", 'n = helpers.measure("abc")'),
    ],
    ids=["from-import", "from-import-alias", "dotted-import", "plain-module"],
)
def test_the_second_run_hits(nb_runner, imports, call):
    work = nb_runner.work_dir
    (work / "helpers.py").write_text(HELPERS, encoding="utf-8")
    (work / "pkg").mkdir()
    (work / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (work / "pkg" / "helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.create_notebook(["import cash\n%cash_on", imports, call])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("n") == "3"
    calls = (work / "calls.log").read_text(encoding="utf-8").split()
    assert calls == ["call"], f"the second run recomputed ({len(calls)} calls)"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("n") == "3"
    calls = (work / "calls.log").read_text(encoding="utf-8").split()
    assert calls == ["call"], f"the run after a restart recomputed ({len(calls)} calls)"
