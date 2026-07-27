"""Two documented behaviours that were wrong, and are now measured.

Both were reported by a reader who did not believe the page, and both times the
reader was right.

**A. `how-it-works/notebook-path.md`** claimed "edit the `AAPL` case and only
that iteration recomputes; `MSFT` and `GOOGL` still hit". `AAPL` is the FIRST
element, and `stats[ticker] = compute(ticker)` accumulates into `stats` — each
iteration reads the dict the previous ones built — so reuse is a *prefix*
property and editing the first entry re-runs all three. The page now states the
prefix rule with this table.

**B. `how-it-works/storage.md`** claimed a "loop body that runs 120 statements
at 0.05 s each ... persists **nothing**". A loop that long is cached as a
*single unit*, and that entry's compute is the whole loop, so it clears the
0.1 s floor and does persist. The claim is true for a flat cell (which is how
`cost-model.md` words it) and for a *short* loop, not for a long one.

Counted, never timed.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

_COUNTER = (
    "import time, pathlib\n"
    "LOG = pathlib.Path(r'{log}')\n"
    "def compute(t):\n"
    "    with LOG.open('a') as fh:\n"
    "        fh.write(str(t) + chr(10))\n"
    "    time.sleep(0.2)\n"          # above the cost-model floor
    "    return len(str(t))\n"
)
_LOOP = ("for ticker in {lst}:\n    stats[ticker] = compute(ticker)\n"
         "print('N', len(stats))")
_BASE = "['AAPL', 'MSFT', 'GOOGL']"


def _n(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


def _recomputed_after(nb_runner, tmp_path, tag, edited):
    """`compute` calls caused by changing the list to *edited*."""
    log = tmp_path / (tag + ".log")
    nb_runner.create_notebook([
        _COUNTER.format(log=log), "stats = {}", _LOOP.format(lst=_BASE),
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 3, "baseline did not run all three iterations"
    nb_runner.set_cell_source(3, _LOOP.format(lst=edited))
    nb_runner.run_cell(3)
    return _n(log) - 3


def test_reuse_is_a_prefix_property(nb_runner, tmp_path):
    """The table on notebook-path.md, row for row."""
    assert _recomputed_after(
        nb_runner, tmp_path, "same", _BASE) == 0, "an unchanged re-run recomputed"
    assert _recomputed_after(
        nb_runner, tmp_path, "append", "['AAPL', 'MSFT', 'GOOGL', 'NVDA']") == 1
    assert _recomputed_after(
        nb_runner, tmp_path, "last", "['AAPL', 'MSFT', 'NVDA']") == 1
    assert _recomputed_after(
        nb_runner, tmp_path, "first", "['AMZN', 'MSFT', 'GOOGL']") == 3, (
        "editing the FIRST entry must re-run all three -- if this drops to 1, "
        "the prefix chain is gone and notebook-path.md should be simplified"
    )


def _loop_badge(nb_runner, iters, body_stmts):
    body = "\n".join(f"    v{k} = ({k} + i) * 2" for k in range(body_stmts))
    nb_runner.create_notebook([
        "%cash_badge print",
        f"import time\nfor i in range({iters}):\n{body}\n    time.sleep(0.005)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    return {
        "single_unit": any("for i in range" in ln for ln in out.splitlines()
                           if "COMPUTED" in ln or "RESTORED" in ln),
        "persisted": "DISK" in out,
        "out": out,
    }


def test_a_short_loop_stays_per_iteration_and_persists_nothing(nb_runner):
    r = _loop_badge(nb_runner, iters=60, body_stmts=1)   # 60 x 2 x 8ms < 1s
    assert not r["single_unit"], f"expected per-iteration rows:\n{r['out'][:400]}"
    assert not r["persisted"], (
        "a loop of individually-cheap statements reached disk; the floor "
        f"warning on storage.md/cost-model.md would be wrong:\n{r['out'][:400]}"
    )


def test_a_long_loop_becomes_one_entry_and_does_persist(nb_runner):
    r = _loop_badge(nb_runner, iters=60, body_stmts=3)   # 60 x 4 x 8ms > 1s
    assert r["single_unit"], (
        "expected the loop to collapse to a single badge row:\n" + r["out"][:400]
    )
    assert r["persisted"], (
        "the single-unit loop did not persist. storage.md says it does -- if "
        f"this is now false, that exception must come out:\n{r['out'][:400]}"
    )
