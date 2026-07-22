"""Adversarial probes: code-review-derived defect hypotheses.

One probe per static-review lead:

 1. test_upstream_reexec_nameerror_swallowed_serves_stale
    checker.py:1354 swallows re-exec errors -> downstream computes on stale x.
 2. test_from_import_virtual_restore_after_restart
    from-import names pollute virtual_modules -> restart restore key diverges.
 3. test_forward_probe_placeholder_earlier_reader
    forward probe plants placeholder; earlier output-less stmt reads it.
 4. test_unsaved_extension_with_user_function_rejected
    _is_valid_extension omits func component -> valid unsaved edit discarded.
 5. test_unrelated_upstream_edit_reruns_loop
    loop-mutated lineage runtime/sim divergence; any upstream edit re-runs loop.
 6. test_empty_cached_value_restore_blocked
    legit-empty cached value can never be restored over non-empty memory.
 7. test_file_touch_reruns_unrelated_loop
    data-file mtime touch conflated with code modification -> loop trust lost.
 8. test_ndarray_iteration_hash_collision_first_run
    sampled ndarray hash + truncated repr -> two iterations share a cache key.
 9. test_fastloop_consuming_iterable_double_eval
    fast-loop heuristic eval()s iterable, single-unit re-evals it (consumed).
10. test_nocache_annotation_on_for_loop
    CacheAnnotation never forwarded to control structures -> no-cache ignored.
11. test_dict_loop_sampled_hash_stale_total
    post-loop lineage = keys-only dict hash -> downstream serves stale sum.
12. test_pathlib_write_text_skipped_on_hit
    Path.write_text not in _WRITE_METHODS -> write cell restored, file missing.
13. test_np_generator_state_diverges_after_edit
    np.random.Generator state untracked -> edited cell draws from wrong state.
14. test_receiver_reset_suppressed_by_name_collision
    CAS-75 suppression keyed by NAME, flow-insensitive -> rebound receiver
    accumulates on isolated re-runs.
"""

import os
import time

import pytest

pytestmark = [pytest.mark.timeout(90)]

REEXEC = "Auto-executing upstream statement:"


def _run_cell_catching(nb_runner, n):
    """Run a cell, returning (raised, exc_text)."""
    try:
        nb_runner.run_cell(n)
        return False, ""
    except Exception as e:  # noqa: BLE001 - CellExecutionError or transport error
        return True, str(e)


# ---------------------------------------------------------------------------
# 1. upstream-reexec-errors-swallowed-stale-state
# ---------------------------------------------------------------------------

def test_upstream_reexec_nameerror_swallowed_serves_stale(nb_runner):
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 2",
        "print('y=' + str(y))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "y=20" in nb_runner.get_output(3)

    # Introduce a NameError typo upstream, save, run only cell 3.
    nb_runner.set_cell_source(1, "x = undefined_name + 1")
    raised, exc_text = _run_cell_catching(nb_runner, 3)
    out = "" if raised else nb_runner.get_output(3)
    raw = exc_text if raised else nb_runner.get_raw_output(3)
    # From-scratch semantics: cell 3 must surface the NameError (or at the
    # very least NOT print the stale y=20 with a success badge).
    surfaced = raised or "NameError" in raw or "undefined_name" in raw
    assert surfaced and "y=20" not in out, (
        f"upstream NameError swallowed; cell3 printed {out!r} (raw head: {raw[:300]!r})"
    )


# ---------------------------------------------------------------------------
# 2. from-import-names-polluting-virtual-modules-key-divergence
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN, cache-effectiveness only (CAS-228). After a restart a chain rooted "
        "in `from math import pi` RE-EXECUTES instead of restoring from disk, "
        "while the `import math` chain beside it restores correctly. The VALUE is "
        "right either way -- this costs time, not correctness -- but from-imports "
        "are ubiquitous, so it is worth fixing rather than pinning as intended. "
        "Not root-caused: the module-component path in cache_key.py contributes "
        "nothing for an unlisted name and so looks stable across restart, meaning "
        "the divergence is elsewhere. strict=True deliberately: the repo's global "
        "xfail_strict is OFF, so without it a fix would land as a silent XPASS."
    ),
)
def test_from_import_virtual_restore_after_restart(nb_runner):
    nb_runner.create_notebook([
        "from math import pi",
        "import math",
        "area = sum(pi * i for i in range(200000))",
        "area2 = sum(math.pi * i for i in range(200000))",
        "print(f'{area:.1f} {area2:.1f}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    first = nb_runner.get_output(5)
    # sum(i for i in range(200000)) * pi = 19_999_900_000 * pi = 62_831_538_912.5
    assert "62831538912" in first, first

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_cell(5)
    out = nb_runner.get_output(5)
    raw = nb_runner.get_raw_output(5)
    assert "62831538912" in out, f"wrong value after restart: {out!r}"
    reexec_from_import = f"{REEXEC} area =" in raw
    reexec_plain_import = f"{REEXEC} area2 =" in raw
    assert not reexec_from_import and not reexec_plain_import, (
        f"post-restart re-execution instead of disk restore: "
        f"from-import chain re-executed={reexec_from_import}, "
        f"plain-import chain re-executed={reexec_plain_import}"
    )


# ---------------------------------------------------------------------------
# 3. forward-probe-placeholder-executed-by-earlier-reader
# ---------------------------------------------------------------------------

def test_forward_probe_placeholder_earlier_reader(nb_runner, tmp_path):
    (tmp_path / "data.csv").write_text("a\n1\n2\n3\n", encoding="utf-8")
    nb_runner.create_notebook([
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
        "print('len=' + str(len(df)))\ndf['c2'] = df['a'].cumsum()",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "len=3" in nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    raised, exc_text = _run_cell_catching(nb_runner, 2)
    out = "" if raised else nb_runner.get_output(2)
    assert not raised and "len=3" in out, (
        f"cell running against forward-probe placeholder? raised={raised} "
        f"exc={exc_text[:300]!r} out={out!r}"
    )


# ---------------------------------------------------------------------------
# 4. is-valid-extension-formula-diverges-from-runtime-lineage
# ---------------------------------------------------------------------------

def test_unsaved_extension_with_user_function_rejected(nb_runner):
    nb_runner.create_notebook([
        "def clean(d):\n    return d.dropna()",
        "import pandas as pd\ndf = pd.DataFrame({'a': [1.0, None, 3.0]})",
        "df = clean(df)",
        "df = df.assign(flag=1)\nprint('idx=' + str(df.index.tolist()))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "idx=[0, 2]" in nb_runner.get_output(4)

    # UNSAVED edit of cell 3 (bypass set_cell_source, which saves the file),
    # then execute it so the kernel holds the extended df.
    nb_runner.nb.cells[2].source = "df = clean(df).reset_index(drop=True)"
    nb_runner.run_cell(3)

    # Re-run cell 4: must compute on the unsaved-extension df (index [0, 1]).
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "idx=[0, 1]" in out, (
        f"unsaved user-function extension discarded; cell4 printed {out!r}"
    )


# ---------------------------------------------------------------------------
# 5. loop-mutation-lineage-formula-divergence-runtime-vs-sim
# ---------------------------------------------------------------------------

def test_unrelated_upstream_edit_reruns_loop(nb_runner):
    nb_runner.create_notebook([
        "base = 1",
        "results = []\nfor i in range(6):\n    results.append(i * i)",
        "print('sum=' + str(sum(results)))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "sum=55" in nb_runner.get_output(3)

    # Genuine but irrelevant upstream edit (cell 3 depends only on results).
    nb_runner.set_cell_source(1, "base = 1\nz = 1")
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    assert "sum=55" in out, out
    loop_rerun = (f"{REEXEC} results = []" in raw) or (f"{REEXEC} for i in" in raw)
    assert not loop_rerun, (
        "unrelated upstream edit re-executed the loop chain "
        f"(markers in raw: {[l for l in raw.splitlines() if REEXEC in l]})"
    )


# ---------------------------------------------------------------------------
# 6. empty-cached-value-restore-blocked
# ---------------------------------------------------------------------------

def test_empty_cached_value_restore_blocked(nb_runner):
    nb_runner.create_notebook([
        "threshold = 100",
        "filtered = [x for x in range(50) if x > threshold]",
        "print('n=' + str(len(filtered)))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "n=0" in nb_runner.get_output(3)

    nb_runner.set_cell_source(1, "threshold = 5")
    nb_runner.run_all()
    assert "n=44" in nb_runner.get_output(3)

    # Flip back: the threshold=100 entry (filtered=[]) exists on disk but
    # memory holds the non-empty threshold=5 list.
    nb_runner.set_cell_source(1, "threshold = 100")
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    assert "n=0" in out, f"WRONG VALUE (worse than lead predicted): {out!r}"
    blocked = "Restore BLOCKED" in raw
    reexec = f"{REEXEC} filtered" in raw
    diag = [l for l in raw.splitlines()
            if "estore" in l or REEXEC in l or "BLOCKED" in l]
    assert not blocked and not reexec, (
        f"legit-empty cached value not restored (blocked={blocked}, "
        f"re-executed={reexec}); restore-related lines: {diag}"
    )


# ---------------------------------------------------------------------------
# 7. file-dep-staleness-conflated-with-code-modification
# ---------------------------------------------------------------------------

def test_file_touch_reruns_unrelated_loop(nb_runner, tmp_path):
    params = tmp_path / "params.json"
    params.write_text('{"k": 1}', encoding="utf-8")
    nb_runner.create_notebook([
        "import json\ncfg = json.load(open('params.json'))",
        "results = []\nfor i in range(6):\n    results.append(i * i)",
        "print('s=' + str(sum(results)) + ' k=' + str(cfg['k']))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "s=55 k=1" in nb_runner.get_output(3)

    # Re-save IDENTICAL content; force an mtime change.
    time.sleep(0.05)
    params.write_text('{"k": 1}', encoding="utf-8")
    st = params.stat()
    os.utime(params, (st.st_atime, st.st_mtime + 2.0))

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    assert "s=55 k=1" in out, out
    loop_rerun = (f"{REEXEC} results = []" in raw) or (f"{REEXEC} for i in" in raw)
    assert not loop_rerun, (
        "data-file mtime touch re-executed the unrelated loop chain "
        f"(markers: {[l for l in raw.splitlines() if REEXEC in l]})"
    )


# ---------------------------------------------------------------------------
# 8. loop-iter-sampled-hash-collision
# ---------------------------------------------------------------------------

def test_ndarray_iteration_hash_collision_first_run(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np\n"
        "batches = [np.zeros(2000), np.zeros(2000)]\n"
        "batches[1][1000] = 5.0",
        "for b in batches:\n"
        "    s = float(b.sum())\n"
        "    print('s=' + str(s))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    # Two iterations differing only in the middle of the array: second must
    # print 5.0, not a replay of iteration 1's cached 0.0.
    assert "s=5.0" in out, f"iteration collision on first run; cell2 printed {out!r}"


# ---------------------------------------------------------------------------
# 9. fastloop-iterable-double-eval
# ---------------------------------------------------------------------------

def test_fastloop_consuming_iterable_double_eval(nb_runner):
    nb_runner.create_notebook([
        "q = list(range(60))",
        "def drain():\n"
        "    global q\n"
        "    out, q = q, []\n"
        "    return out",
        "total = 0\n"
        "for item in drain():\n"
        "    a = item + 1\n"
        "    b = a * 2\n"
        "    total = total + b",
        "print('total=' + str(total))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(4)
    # sum((i+1)*2 for i in range(60)) == 3660; double-evaluating drain()
    # consumes q before the loop body ever runs -> total stays 0.
    assert "total=3660" in out, f"consuming iterable evaluated twice: {out!r}"


# ---------------------------------------------------------------------------
# 10. annotations-dropped-in-control-structures
# ---------------------------------------------------------------------------

def test_nocache_annotation_on_for_loop(nb_runner):
    nb_runner.create_notebook([
        "import time",
        "# @cash:no-cache\n"
        "for i in range(3):\n"
        "    t = time.perf_counter()\n"
        "    print('t' + str(i) + '=' + format(t, '.9f'))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    first = nb_runner.get_output(2)
    assert "t0=" in first and "t2=" in first, first

    nb_runner.run_cell(2)
    second = nb_runner.get_output(2)
    assert "t0=" in second and "t2=" in second, second
    # no-cache promises plain re-execution -> perf_counter values MUST differ.
    assert first != second, (
        f"@cash:no-cache ignored on for-loop: identical timestamps replayed "
        f"({first!r})"
    )


# ---------------------------------------------------------------------------
# 11. loop-lineage-sampled-dict-stale
# ---------------------------------------------------------------------------

def test_dict_loop_sampled_hash_stale_total(nb_runner):
    nb_runner.create_notebook([
        "d = {i: 1.0 for i in range(300)}",
        "factor = 2.0",
        "for k in d:\n    d[k] = d[k] * factor",
        "total = sum(d.values())\nprint('total=' + str(total))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "total=600.0" in nb_runner.get_output(4)

    nb_runner.set_cell_source(2, "factor = 3.0")
    nb_runner.run_all()
    out = nb_runner.get_output(4)
    assert "total=900.0" in out, (
        f"stale cached total served after factor edit (keys-only dict hash): {out!r}"
    )


# ---------------------------------------------------------------------------
# 12. write-side-effect-gaps (pathlib write_text)
# ---------------------------------------------------------------------------

def test_pathlib_write_text_skipped_on_hit(nb_runner, tmp_path):
    nb_runner.create_notebook([
        "from pathlib import Path\ncfg = {'x': 1}",
        "nchars = Path('out.json').write_text(str(cfg))",
        "print('content=' + open('out.json').read())",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out_file = tmp_path / "out.json"
    assert out_file.exists(), "setup: write cell did not create the file"
    assert "content={'x': 1}" in nb_runner.get_output(3)

    out_file.unlink()
    raised, exc_text = _run_cell_catching(nb_runner, 2)
    assert not raised, exc_text[:300]
    assert out_file.exists(), (
        "Path.write_text cell served from cache without executing: "
        "out.json was NOT recreated"
    )


# ---------------------------------------------------------------------------
# 13. generator-rng-untracked
# ---------------------------------------------------------------------------

def test_np_generator_state_diverges_after_edit(nb_runner):
    c1 = "import numpy as np\nrng = np.random.default_rng(0)"
    c2 = "a = rng.normal(size=3)\nprint('a=' + ','.join(format(v, '.5f') for v in a))"
    c3 = "b = rng.normal(size=3)\nprint('b=' + ','.join(format(v, '.5f') for v in b))"
    c3_edited = "b = rng.normal(size=4)\nprint('b=' + ','.join(format(v, '.5f') for v in b))"

    # Ground truth: fresh from-scratch run of the EDITED notebook, no cash.
    nb_runner.create_notebook([c1, c2, c3_edited])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    gt_lines = [l for l in nb_runner.get_output(3).splitlines() if l.startswith("b=")]
    assert gt_lines, nb_runner.get_output(3)
    gt_b = gt_lines[0]
    nb_runner.shutdown()

    # Cash session: run, re-run (cache pass), then edit cell 3 and run it.
    nb_runner.create_notebook([c1, c2, c3])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, c3_edited)
    nb_runner.run_cell(3)
    out_lines = [l for l in nb_runner.get_output(3).splitlines() if l.startswith("b=")]
    assert out_lines, nb_runner.get_output(3)
    # ADJUDICATED: the documented per-object-generator limitation, kept as-is by
    # explicit decision (docs/known-limitations.md, "Per-object generators
    # (np.random.default_rng) are only partially tracked" — facet 1, stream
    # position across cells).
    #
    # A per-object generator's POSITION is not tracked: `rng` is one live object
    # carrying hidden state, and an isolated re-run draws from wherever the
    # previous run left it rather than the position it would hold top-to-bottom.
    # Fixing it needs per-object stream tracking, which numpy's C types block —
    # a Generator is not attr-settable or weakref-able and its methods cannot be
    # wrapped, so there is nowhere to hang the bookkeeping. The module-global
    # channel IS position-aware (test_rng_position_aware.py), which is exactly
    # why the docs point there as the remedy.
    #
    # Pinned as the real behaviour so the limitation stays visible; if per-object
    # tracking ever lands, this test fails and says so.
    assert out_lines[0] != gt_b, (
        "per-object generator position now matches a from-scratch run — the "
        "documented limitation appears to be FIXED; update known-limitations.md "
        "and turn this into an equality assertion"
    )


# ---------------------------------------------------------------------------
# 14. op-reset-suppression-name-keyed
# ---------------------------------------------------------------------------

def test_receiver_reset_suppressed_by_name_collision(nb_runner):
    nb_runner.create_notebook([
        "class Stack:\n"
        "    def __init__(self):\n"
        "        self.items = []\n"
        "    def push(self, x):\n"
        "        self.items.append(x)",
        "s = Stack()\ns.push(1)",
        "s = Stack()",
        "s.push(2)\nprint('n=' + str(len(s.items)))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "n=1" in nb_runner.get_output(4)

    for attempt in (1, 2, 3):
        nb_runner.run_cell(4)
        out = nb_runner.get_output(4)
        assert "n=1" in out, (
            f"isolated re-run #{attempt} accumulated (receiver reset suppressed "
            f"by same-name mutation in another cell): {out!r}"
        )
