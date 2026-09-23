"""After a restart, what a loop built is restored downstream, not rebuilt by replaying the loop.

Seen in two users' notebooks: a restart and one run of the last cell
replayed every loop above it -- one re-read a 1,312-file folder, the other re-ran
a 263 s parameter sweep. A loop's outputs get lineages the runtime derives
from the VALUES it built; the simulation cannot derive them from code, and the
session's record of them (``TrackingState.control_outcomes``) died with the
kernel. So after a restart the simulated key of anything computed from a loop
never matched its entry, and the planner rebuilt the loop's outputs to run it.

The record is persisted now -- but only for a loop whose outcome is all it did,
and trusted only while everything its callees read is unchanged. A record
trusted wrongly skips a loop that should have run, so most of this file is
about loops that must NOT be trusted.

Counted, not timed: a tee on ``StatementProcessor.process_statement`` in the
new kernel records which statements ran.
"""

import ast
import os
from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

N = 40

_TEE = """
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if not hasattr(C, "_test_orig"):
    C._test_orig = C.process_statement
    def _tee(self, code, *a, **k):
        r = C._test_orig(self, code, *a, **k)
        try:
            s = r.get("status")
            if str(getattr(s, "value", s)) == "COMPUTED":
                C._test_ran.append(str(code).strip().splitlines()[-1].strip()[:40])
        except Exception:
            pass
        return r
    C.process_statement = _tee
C._test_ran = []
"""
_UNTEE = """
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if hasattr(C, "_test_orig"):
    C.process_statement = C._test_orig
    del C._test_orig
"""
_RAN = "__import__('cash.notebook.statement.processor', fromlist=['_']).StatementProcessor._test_ran"


def _record_expr(loop_src: str) -> str:
    """The persisted record of *loop_src*'s loop, as the kernel's backend holds it."""
    import ast as _ast

    code = _ast.unparse(_ast.parse(loop_src).body[1])  # body[0] is the accumulator seed
    return (
        "__import__('cash')._global_cash.backend.get_metadata("
        f"__import__('cash.notebook.cache_key', fromlist=['_']).control_outcome_key({code!r}))"
    )


SETUP = (
    "import glob\nimport os\nimport time\nimport pandas as pd\n"
    "files = sorted(glob.glob('exports/*.csv'))\n"
    "def parse(f):\n    d = pd.read_csv(f)\n    d['v'] = clean(d['v'])\n    return d\n"
    "def slow_summary(t):\n    time.sleep(0.3)\n    return int(t['v'].sum())"
)
#: Defined BELOW ``parse``, which calls it, so no lineage the loop's inputs
#: carry depends on it: only the record's callee lineages see it change.
CLEAN = "def clean(v):\n    return v"
LOOP = "parts = []\nfor f in files:\n    d = parse(f)\n    parts.append(d)\nraw = pd.concat(parts, ignore_index=True)"
SUMMARY = "summary = slow_summary(raw)"
REPORT = "print('S', summary)"
READS_PARTS = "print('P', len(parts), len(d))"
CELLS = ["import cash\n%cash_on", SETUP, CLEAN, LOOP, SUMMARY, REPORT, READS_PARTS]
REPORT_CELL, PARTS_CELL = 6, 7
WANT = f"S {sum(range(50)) * N}"


@pytest.fixture
def _teed(nb_runner):
    yield
    try:
        nb_runner.peek(f"exec({_UNTEE!r}, {{}})")
    except Exception:  # noqa: BLE001 - a kernel that never started has nothing to undo
        pass


def _files(work):
    folder = Path(work) / "exports"
    folder.mkdir()
    for i in range(N):
        path = folder / f"e{i:03d}.csv"
        path.write_text("v\n" + "\n".join(str(j) for j in range(50)) + "\n", encoding="utf-8")
        st = os.stat(path)
        os.utime(path, (st.st_atime - 3600, st.st_mtime - 3600))


def _restart_and_run(nb_runner, cell: int) -> list[str]:
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    # The peek's own upstream check simulated the notebook and cached it
    # before the tee existed; %cash_on drops that cache.
    nb_runner.peek("get_ipython().run_line_magic('cash_on', '')")
    nb_runner.run_cell(cell)
    return [c for c in ast.literal_eval(nb_runner.peek(_RAN)) if "__CASH_PEEK__" not in c]


def _run_all(nb_runner, cells):
    _files(nb_runner.work_dir)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()


def test_a_restart_restores_what_a_loop_built_without_replaying_it(nb_runner, _teed):
    _run_all(nb_runner, CELLS)
    assert WANT in nb_runner.get_output(REPORT_CELL)
    assert nb_runner.peek(_record_expr(LOOP)) != "None", "the loop's outcome was not persisted"

    ran = _restart_and_run(nb_runner, REPORT_CELL)

    assert WANT in nb_runner.get_output(REPORT_CELL)
    # Measured before the fix: every `d = parse(f)`, the concat and the setup
    # cell ran again, and only then did `summary` restore.
    assert ran == [REPORT], f"ran after the restart: {ran}"


#: Makes the loop's first run record a split verdict whatever the machine's
#: load. The verdict is a wall-clock judgement of the first iterations
#: (``LoopSplitPolicy``): about 3 to 6 ms each records one, so on a busy
#: machine this notebook learned a split about one run in six, and with it the
#: restart replayed everything. Raising the ceiling and dropping the floor makes
#: that case the one every run takes.
LEARNS_A_SPLIT = "cash.configure(loop_split_max_iter_seconds=1.0, loop_split_min_remaining_seconds=0.0)"


def _split_expr(loop_src: str) -> str:
    """The split point recorded for *loop_src*'s loop, as the kernel's store holds it."""
    code = ast.unparse(ast.parse(loop_src).body[1])
    split = "__import__('cash.notebook.loop_split', fromlist=['_'])"
    return (
        f"{split}.store_for_backend(__import__('cash')._global_cash.backend)"
        f".get({split}.loop_source_hash(__import__('ast').parse({code!r}).body[0]))"
    )


def test_a_restart_restores_what_a_loop_built_after_it_learned_to_split(nb_runner, _teed):
    """A split verdict applies from the loop's NEXT run, so the run that
    learned it ran the loop whole and recorded its outcome under the whole
    loop. Modelled as a head and a tail after the restart, the loop had no
    outcome to restore from, and every cell above the report ran again."""
    _run_all(nb_runner, [f"{CELLS[0]}\n{LEARNS_A_SPLIT}", *CELLS[1:]])
    assert WANT in nb_runner.get_output(REPORT_CELL)
    assert nb_runner.peek(_split_expr(LOOP)) != "None", "the loop learned no split"
    assert nb_runner.peek(_record_expr(LOOP)) != "None", "the loop's outcome was not persisted"

    ran = _restart_and_run(nb_runner, REPORT_CELL)

    assert WANT in nb_runner.get_output(REPORT_CELL)
    assert ran == [REPORT], f"ran after the restart: {ran}"


def test_a_loop_that_learned_to_split_still_sees_a_changed_file(nb_runner, _teed):
    """Negative control for the test above: the whole loop's outcome is used
    only while it still holds. A file behind it changed, so the split loop runs."""
    _run_all(nb_runner, [f"{CELLS[0]}\n{LEARNS_A_SPLIT}", *CELLS[1:]])
    assert nb_runner.peek(_split_expr(LOOP)) != "None", "the loop learned no split"
    edited = Path(nb_runner.work_dir) / "exports" / "e000.csv"
    edited.write_text("v\n" + "\n".join(str(j + 1) for j in range(50)) + "\n", encoding="utf-8")

    ran = _restart_and_run(nb_runner, REPORT_CELL)

    assert f"S {sum(range(50)) * N + 50}" in nb_runner.get_output(REPORT_CELL), nb_runner.get_output(REPORT_CELL)
    assert "d = parse(f)" in ran, f"ran after the restart: {ran}"


def test_a_helper_edited_through_another_helper_is_not_served_stale(nb_runner, _teed):
    """The loop calls ``parse``, which calls ``clean``. Editing ``clean`` changes
    no lineage the loop's own inputs name -- only its callees' globals do."""
    _run_all(nb_runner, CELLS)
    nb_runner.set_cell_source(3, CLEAN.replace("return v", "return v + 1"))

    _restart_and_run(nb_runner, REPORT_CELL)

    out = nb_runner.get_output(REPORT_CELL)
    assert f"S {(sum(range(50)) + 50) * N}" in out, out


def test_the_loops_own_variables_are_rebuilt_when_a_cell_reads_them(nb_runner, _teed):
    """``parts`` and ``d`` are not stored anywhere a restart can restore them from."""
    _run_all(nb_runner, CELLS)

    ran = _restart_and_run(nb_runner, PARTS_CELL)

    assert f"P {N} 50" in nb_runner.get_output(PARTS_CELL)
    assert "d = parse(f)" in ran


@pytest.mark.parametrize(
    "why, loop",
    [
        (
            "writes a file",
            "parts = []\nfor f in files:\n    d = parse(f)\n    open(f + '.seen', 'w').write('x')\n    parts.append(d)",
        ),
        (
            "draws from the global RNG",
            "parts = []\nfor f in files:\n    d = parse(f)\n    d['r'] = random.random()\n    parts.append(d)",
        ),
        (
            "reads the clock",
            "parts = []\nfor f in files:\n    d = parse(f)\n    d['t'] = time.time()\n    parts.append(d)",
        ),
        (
            "has a helper that mutates a global",
            "parts = []\nfor f in files:\n    d = parse(f)\n    note(f)\n    parts.append(d)",
        ),
    ],
)
def test_a_loop_that_did_more_than_build_its_outputs_is_not_recorded(nb_runner, _teed, why, loop):
    setup = SETUP + "\nimport random\nSEEN = []\ndef note(f):\n    SEEN.append(f)"
    _run_all(nb_runner, ["import cash\n%cash_on", setup, CLEAN, loop + "\nraw = pd.concat(parts)", SUMMARY])

    assert nb_runner.peek(_record_expr(loop)) == "None", f"a loop that {why} was recorded"
