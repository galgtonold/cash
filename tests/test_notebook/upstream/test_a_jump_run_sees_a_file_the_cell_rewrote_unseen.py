"""A run below a write cash does not see as one simulates the file as rewritten.

A cell writes a file through something no ``open`` event reports (another
process, SQLite, h5py's C library, C stdio through ctypes) and then reads it.
On a run that jumps over the cell, ``plan_cell_run`` simulated the read with
the file component the run had kept from before the write: the lineage of
the cell's last version came out as the previous run's. Two guards now each
keep it right: the plan looks at each file again, and the statement that
wrote is taken for a writer, which drops the kept answers.
"""

from __future__ import annotations

import ctypes
import sys

import pytest

from cash.analysis.namespace_effects import statement_calls_through_ctypes
from tests._cell_driver import run_cash_cell

SETUP = """
import subprocess, sqlite3, sys
"""

LIBC_SETUP = """
import ctypes
_libc = ctypes.CDLL(None)
_libc.fopen.restype = ctypes.c_void_p
def _libc_write(path, data):
    fp = _libc.fopen(path, b"w")
    _libc.fwrite(data, 1, len(data), ctypes.c_void_p(fp))
    _libc.fclose(ctypes.c_void_p(fp))
"""

WRITERS = {
    "subprocess": "subprocess.run([sys.executable, '-c', "
    "'import sys; open(sys.argv[2], \"w\").write(sys.argv[1])', VAL, F], check=True)",
    "ctypes": "_libc_write(F.encode(), VAL.encode())",
    "sqlite": "_c = sqlite3.connect(F); _c.execute('create table if not exists t(v)'); "
    "_c.execute('insert into t values (?)', (VAL * 2000,)); _c.commit(); _c.close()",
    "h5py": "import h5py\n_h = h5py.File(F, 'w'); _h['v'] = VAL; _h.close()",
}


@pytest.mark.parametrize("plan_looks_again", [True, False], ids=["both guards", "writer check alone"])
@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_the_plan_sees_the_file_the_cell_just_rewrote(
    cash_magics, mock_shell, tmp_path, monkeypatch, writer, plan_looks_again
):
    if writer == "h5py":
        pytest.importorskip("h5py")
    if writer == "ctypes" and sys.platform == "win32":
        pytest.skip("C stdio through ctypes.CDLL(None) is POSIX only")
    from cash.notebook.upstream import simulator
    from cash.notebook.upstream.virtual_lineage import VirtualLineage

    if not plan_looks_again:
        monkeypatch.setattr(simulator, "forget_file_state_this_run", lambda: None)
    simulated, plans = [], []
    real_simulate = VirtualLineage.simulate_one_node

    def simulate_one_node(self, sim, *args, **kwargs):
        before = len(sim.trace)
        result = real_simulate(self, sim, *args, **kwargs)
        simulated.extend(sim.trace[before:])
        return result

    real_plan = simulator.NotebookSimulator.plan_cell_run

    def plan_cell_run(self, *args, **kwargs):
        recorded = self.tracking_state.variable_lineage.get("x")
        del simulated[:]
        result = real_plan(self, *args, **kwargs)
        plans.append((recorded, simulated[-1].produced_lineages.get("x") if simulated else None))
        return result

    monkeypatch.setattr(VirtualLineage, "simulate_one_node", simulate_one_node)
    monkeypatch.setattr(simulator.NotebookSimulator, "plan_cell_run", plan_cell_run)

    path = tmp_path / "data.bin"
    path.write_text("" if writer == "sqlite" else "v0", encoding="utf-8")
    run_cash_cell(cash_magics, SETUP + (LIBC_SETUP if writer == "ctypes" else ""))
    mock_shell.user_ns["F"] = str(path)
    above = "a = open(F, 'rb').read()"
    cell = WRITERS[writer] + "\nx = open(F, 'rb').read()\nx = x.upper()\nx = x + b'!'"
    cells = [above, cell]
    for i in range(1, 4):
        mock_shell.user_ns["VAL"] = f"v{i}" * i
        run_cash_cell(cash_magics, above, cells=cells)
        run_cash_cell(cash_magics, cell, cells=cells)
        assert mock_shell.user_ns["x"] == path.read_bytes().upper() + b"!"
    stale = [n for n, (recorded, final) in enumerate(plans) if recorded is not None and recorded == final]
    assert not stale, f"the plan simulated the lineage from before the write in runs {stale}"


@pytest.mark.skipif(sys.platform == "win32", reason="ctypes.CDLL(None) is POSIX only")
def test_a_call_through_a_ctypes_library_is_taken_for_a_write():
    libc = ctypes.CDLL(None)
    ns = {"libc": libc, "getpid": libc.getpid, "ctypes": ctypes, "CDLL": ctypes.CDLL}
    assert statement_calls_through_ctypes("n = libc.fwrite(b, 1, 3, fp)", ns) == "libc.fwrite"
    assert statement_calls_through_ctypes("pid = getpid()", ns) == "getpid"
    assert statement_calls_through_ctypes("ctypes.memmove(dst, src, 3)", ns) == "ctypes.memmove"
    assert statement_calls_through_ctypes("lib = CDLL('libm.so.6')", ns) == "CDLL"


def test_a_statement_with_no_ctypes_call_is_not_one():
    ns = {"ctypes": ctypes, "len": len, "lib": object()}
    assert statement_calls_through_ctypes("n = len(xs)", ns) is None
    assert statement_calls_through_ctypes("n = lib.size()", ns) is None
    assert statement_calls_through_ctypes("ctypes", ns) is None
