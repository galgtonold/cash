"""Integration tests for CAS-99: file-dep staleness must not disable loop trust.

A changed (or merely mtime-touched) file dependency used to set a GLOBAL
"cache had a hash mismatch" flag, conflating file staleness with code
modification. That flag becomes ``upstream_has_modifications``, which disables
the loop-derived TRUST mechanism for EVERY loop var in the notebook -- so a
completely unrelated loop chain would be spuriously re-executed (its runtime
value-hash lineage and the simulator's input-lineage fold are structurally
divergent and can never agree, so the raw comparison always "mismatches").

The fix decouples the two: a file change still re-simulates from the reader
cell so ``vars_with_stale_files`` (the reader's own dedicated invalidation
channel) is repopulated, but it no longer sets the global mismatch flag, so
unrelated loop chains keep their in-memory trust.

The CRITICAL guard here is ``test_changed_file_feeding_loop_reruns``: a loop
that GENUINELY consumes the file's data MUST still re-execute when the file
changes. If that regresses, the fix under-invalidates.
"""
import json
import os
import time

import pytest

pytestmark = [pytest.mark.files, pytest.mark.loops, pytest.mark.timeout(90)]

REEXEC = "Auto-executing upstream statement:"


def _loop_reran(raw: str) -> bool:
    """True if the ``results``/``acc`` loop chain re-executed per cash's markers."""
    return (
        f"{REEXEC} results = []" in raw
        or f"{REEXEC} for i in" in raw
        or f"{REEXEC} acc = []" in raw
        or f"{REEXEC} for v in" in raw
    )


def _markers(raw: str) -> list[str]:
    return [line for line in raw.splitlines() if REEXEC in line]


def test_file_touch_does_not_rerun_unrelated_loop(nb_runner, tmp_path):
    """FIX (touch): an mtime touch of an UNRELATED file must not re-run a loop.

    The loop (``results``) never reads ``params.json``; only cell 1 does. A pure
    ``os.utime`` touch (identical content) of the json must NOT re-execute the
    loop chain.
    """
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

    # Re-save IDENTICAL content, then force an mtime change (pure touch).
    time.sleep(0.05)
    params.write_text('{"k": 1}', encoding="utf-8")
    st = params.stat()
    os.utime(params, (st.st_atime, st.st_mtime + 2.0))

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    assert "s=55 k=1" in out, out
    assert not _loop_reran(raw), (
        "mtime touch of an unrelated file re-executed the loop chain "
        f"(markers: {_markers(raw)})"
    )


def test_changed_unrelated_file_does_not_rerun_loop(nb_runner, tmp_path):
    """FIX (changed content): a genuine change to an UNRELATED file re-runs only
    its reader, never the loop.

    ``params.json``'s content changes (k: 1 -> 2), so cell 1's reader re-runs and
    the new ``k`` is reflected -- but the loop, which does not read the json, must
    NOT re-execute.
    """
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

    # Genuinely change the json CONTENT (the loop does not depend on it).
    time.sleep(0.05)
    params.write_text('{"k": 2}', encoding="utf-8")

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    # New file data must be reflected (the reader re-ran)...
    assert "s=55 k=2" in out, out
    # ...but the unrelated loop chain must NOT have re-executed.
    assert not _loop_reran(raw), (
        "changed unrelated file re-executed the loop chain "
        f"(markers: {_markers(raw)})"
    )


def test_changed_file_feeding_loop_reruns(nb_runner, tmp_path):
    """UNDER-INVALIDATION GUARD: a loop that CONSUMES the file's data MUST re-run
    when the file changes, and reflect the new data.

    ``data`` is read from ``params.json``; the loop derives ``acc`` from ``data``.
    Changing the json content must re-execute the loop chain and produce the
    ground-truth output for the new data. If this fails, the fix under-invalidates
    (file staleness no longer reaches a loop that legitimately depends on it).
    """
    params = tmp_path / "params.json"
    params.write_text('{"vals": [1, 2, 3]}', encoding="utf-8")
    nb_runner.create_notebook([
        "import json",
        "data = json.load(open('params.json'))['vals']",
        "acc = []\nfor v in data:\n    acc.append(v * 2)",
        "print('acc=' + str(acc))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    # Ground truth for [1, 2, 3]: each doubled.
    assert "acc=[2, 4, 6]" in nb_runner.get_output(4)

    # Change the json CONTENT that the loop genuinely consumes.
    time.sleep(0.05)
    params.write_text('{"vals": [10, 20]}', encoding="utf-8")

    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    raw = nb_runner.get_raw_output(4)
    # Ground truth for the NEW data [10, 20]: each doubled.
    assert "acc=[20, 40]" in out, (
        "loop consuming changed file data served STALE result (under-invalidation): "
        f"{out!r}"
    )
    # And the loop chain must have actually re-executed to produce it.
    assert _loop_reran(raw), (
        "loop consuming changed file data did not re-execute per cash's markers "
        f"(markers: {_markers(raw)})"
    )
