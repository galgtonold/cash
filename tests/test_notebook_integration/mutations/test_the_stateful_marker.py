"""Adversarial probes: @stateful marker.

 1. test_stateful_marker_forces_reexecution — a @stateful-marked function's
        call cell must re-execute on an unchanged run_all (plain-kernel
        semantics: the hidden counter increments).

(A %cash_clear probe was removed: no such magic exists.)
"""

import pytest

pytestmark = [pytest.mark.timeout(120)]

MISS = "Executing (cache miss)"


def test_stateful_marker_forces_reexecution(nb_runner):
    nb_runner.create_notebook(
        [
            "from cash.purity import stateful\n"
            "calls6 = []\n"
            "@stateful\n"
            "def next_id():\n"
            "    calls6.append(1)\n"
            "    return len(calls6)",
            "nid = next_id()\nprint('nid=', nid)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "nid= 1" in nb_runner.get_output(2)

    nb_runner.run_all()
    out = nb_runner.get_output(2)
    # Plain-kernel semantics for a stateful call on a full rerun: increments.
    # (Function cell re-runs too, resetting calls6 -> next_id() returns 1 again;
    # so the observable invariant is simply: no crash and nid consistent with a
    # genuine re-execution, i.e. 1 — the cell must NOT be skipped into printing
    # a stale replay marked as cached.)
    assert "nid= 1" in out, f"stateful rerun output wrong: {out!r}"
    raw = nb_runner.get_raw_output(2)
    assert MISS in raw, "@stateful call cell was served from cache / skipped on rerun (marker promises re-execution)"


# A @stateful helper in the user's own module, as after "Moving to a module".
# It counts its calls with os.open/os.write, which a cache hit does not replay
# and cash does not take for a file write, and sleeps past the store floor so
# an unrefused statement would really be stored.
HELPERS = """import os
import time
from cash import stateful

@stateful
def announce(text):
    time.sleep(0.2)
    fd = os.open("sent.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.write(fd, (text + "\\n").encode())
    os.close(fd)
    return 1
"""


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        ("import helpers", 'receipt = helpers.announce("trained")'),
        ("from pkg import helpers", 'receipt = helpers.announce("trained")'),
        ("import pkg.helpers", 'receipt = pkg.helpers.announce("trained")'),
        ("import helpers", 'receipt = str(helpers.announce("trained"))'),
        ("from helpers import announce", 'receipt = announce("trained")'),
    ],
    ids=["module", "package", "package-dotted", "nested-in-expression", "from-import"],
)
def test_a_stateful_function_in_a_module_runs_every_time(nb_runner, imports, call):
    """Nothing is edited; every run must call the function again. Three runs,
    not two: the first key of a statement reading a package submodule differs
    from the later ones, so the second run alone would miss either way."""
    work = nb_runner.work_dir
    (work / "helpers.py").write_text(HELPERS, encoding="utf-8")
    (work / "pkg").mkdir()
    (work / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (work / "pkg" / "helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.create_notebook(["import cash\n%cash_on", imports, call])
    nb_runner.start_kernel()
    for _ in range(3):
        nb_runner.run_all()
    sent = (work / "sent.log").read_text(encoding="utf-8").split()
    assert sent == ["trained"] * 3, "the @stateful call was served from the cache"
