"""CAS-81/82: file writes participate in upstream re-execution.

File writes have no variable edge, so the backward scan never scheduled
them: editing a writer cell and re-running only the reader served the stale
pre-edit file (CAS-81), and even when the writer's sibling statements DID
re-run, the side-effect-only write statement was skipped and the reader's
freshness stayed decided against the pre-run file state (CAS-82).
"""

import pytest

pytestmark = [pytest.mark.timeout(120), pytest.mark.files]

REEXEC = "Auto-executing upstream statement:"


def _p(path) -> str:
    return str(path).replace("\\", "/")


def test_writer_edit_isolated_reader_sees_new_file(nb_runner, tmp_path):
    p = _p(tmp_path / "data.txt")
    nb_runner.create_notebook([
        f"with open('{p}', 'w') as f:\n    f.write('1,2,3')\nprint('wrote v1')",
        f"with open('{p}') as f:\n    body = f.read()\nprint('body =', body)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "body = 1,2,3" in nb_runner.get_output(2)

    nb_runner.set_cell_source(
        1, f"with open('{p}', 'w') as f:\n    f.write('10,20,30,40')\nprint('wrote v2')"
    )
    nb_runner.run_cell(2)
    assert "body = 10,20,30,40" in nb_runner.get_output(2), (
        "reader served stale file content after the writer cell was edited"
    )


def test_bare_tocsv_writer_edit_isolated_downstream(nb_runner, tmp_path):
    """Bare expression writers (no outputs) need their own trace entry."""
    p = _p(tmp_path / "made.csv")
    nb_runner.create_notebook([
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [1, 2, 3]}}).to_csv('{p}', index=False)\n"
        "print('wrote v1')",
        f"import pandas as pd\ndf = pd.read_csv('{p}')\nprint('vals =', df['v'].tolist())",
        "print('total =', int(df['v'].sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total = 6" in nb_runner.get_output(3)

    nb_runner.set_cell_source(
        1,
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [10, 20, 30, 40]}}).to_csv('{p}', index=False)\n"
        "print('wrote v2')",
    )
    nb_runner.run_cell(3)
    assert "total = 100" in nb_runner.get_output(3), (
        "downstream served a stale total after the to_csv writer was edited"
    )


def test_midrun_write_reader_freshness_post_write(nb_runner, tmp_path):
    """CAS-82: the re-executed writer must actually write, and the reader's
    freshness must be decided AFTER that write — no impossible namespace."""
    p = _p(tmp_path / "both.pkl")
    writer_v1 = (
        "import pickle\n"
        "payload = {'nums': [1, 2, 3]}\n"
        f"with open('{p}', 'wb') as f:\n"
        "    pickle.dump(payload, f)\n"
        "print('dumped', len(payload['nums']))"
    )
    nb_runner.create_notebook([
        writer_v1,
        f"import pickle\nwith open('{p}', 'rb') as f:\n    loaded = pickle.load(f)\nprint('loaded =', loaded['nums'])",
        "print('combo =', sum(loaded['nums']), len(payload['nums']))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "combo = 6 3" in nb_runner.get_output(3)

    nb_runner.set_cell_source(1, writer_v1.replace("[1, 2, 3]", "[10, 20, 30, 40]"))
    nb_runner.run_cell(3)
    assert "combo = 100 4" in nb_runner.get_output(3), (
        "payload/loaded diverged: reader freshness was pre-decided against "
        "the pre-write file state"
    )


def test_pickle_writer_edit_when_only_reader_var_needed(nb_runner, tmp_path):
    """The write may be the only consumer of the edited payload: the writer
    (unchanged code) must still re-run because its INPUT lineage drifted."""
    p = _p(tmp_path / "payload.pkl")
    writer_v1 = (
        "import pickle\n"
        "payload = {'nums': [1, 2, 3]}\n"
        f"with open('{p}', 'wb') as f:\n"
        "    pickle.dump(payload, f)\n"
        "print('dumped', len(payload['nums']))"
    )
    nb_runner.create_notebook([
        writer_v1,
        f"import pickle\nwith open('{p}', 'rb') as f:\n    loaded = pickle.load(f)\nprint('loaded =', loaded['nums'])",
        "print('lsum =', sum(loaded['nums']))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "lsum = 6" in nb_runner.get_output(3)

    nb_runner.set_cell_source(1, writer_v1.replace("[1, 2, 3]", "[10, 20, 30, 40]"))
    nb_runner.run_cell(3)
    assert "lsum = 100" in nb_runner.get_output(3), (
        "edited payload never reached the file: writer with unchanged code "
        "was not re-scheduled"
    )


def test_unchanged_rerun_does_not_rerun_writer(nb_runner, tmp_path):
    """EFFECTIVENESS control: an unchanged isolated re-run must not
    re-execute the (already-executed) writer."""
    p = _p(tmp_path / "stable.txt")
    nb_runner.create_notebook([
        f"with open('{p}', 'w') as f:\n    f.write('same')\nprint('wrote')",
        f"with open('{p}') as f:\n    body2 = f.read()\nprint('body2 =', body2)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "body2 = same" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    raw = nb_runner.get_raw_output(2)
    assert "body2 = same" in out
    assert REEXEC not in raw, (
        f"unchanged rerun re-executed upstream statements: "
        f"{[l for l in raw.splitlines() if REEXEC in l]}"
    )


def test_unrelated_edit_does_not_rerun_writer(nb_runner, tmp_path):
    """EFFECTIVENESS control: editing a cell unrelated to the writer must
    not re-execute the writer."""
    p = _p(tmp_path / "stable2.txt")
    nb_runner.create_notebook([
        "tag = 'a'",
        f"with open('{p}', 'w') as f:\n    f.write('fixed')\nprint('wrote')",
        f"with open('{p}') as f:\n    body3 = f.read()\nprint('body3 =', body3, tag)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "body3 = fixed a" in nb_runner.get_output(3)

    nb_runner.set_cell_source(1, "tag = 'b'")
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    assert "body3 = fixed b" in out, out
    writer_reruns = [
        l for l in raw.splitlines() if REEXEC in l and "open(" in l and "'w'" in l
    ]
    assert not writer_reruns, (
        f"unrelated edit re-executed the writer: {writer_reruns}"
    )


def test_writer_input_restored_after_kernel_restart(nb_runner, tmp_path):
    """CAS-153: after a REAL kernel restart, running only a downstream file
    reader must not crash.

    Post-restart ``executed_write_stmt_codes`` is empty, so the bare
    ``df.to_csv(path)`` writer looks stale and is scheduled — but its input
    ``df`` is gone from the restarted namespace. The scheduler used to read
    "no lineage either side" as "unchanged" and never scheduled df's producer,
    so the writer ran against a missing ``df`` and raised NameError, which was
    re-raised into whatever cell the user ran and poisoned the whole notebook.
    The producer must be re-materialised (cache-restored or recomputed) first.
    """
    p = _p(tmp_path / "sales.csv")
    nb_runner.create_notebook([
        "import cash\nimport pandas as pd",                                  # 1: imports
        "%cash_on",                                                          # 2: enable
        "df = pd.DataFrame({'v': list(range(1000))})\nprint('built', len(df))",  # 3: producer
        f"df.to_csv('{p}', index=False)\nprint('wrote')",                    # 4: bare writer
        f"df2 = pd.read_csv('{p}')\nprint('sum =', int(df2['v'].sum()))",    # 5: reader
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "sum = 499500" in nb_runner.get_output(5)

    # ACTUAL kernel restart — clears user_ns, so df is genuinely absent.
    import asyncio
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)

    # Run ONLY the reader. Upstream must re-materialise df and re-run the writer
    # instead of exec-ing df.to_csv against a missing name.
    nb_runner.run_cell(5)
    out = nb_runner.get_output(5)
    assert "NameError" not in out and "UpstreamStateError" not in out, (
        f"writer re-fired against a missing df, poisoning the notebook: {out!r}"
    )
    assert "sum = 499500" in out, (
        f"reader crashed or served wrong data after restart: {out!r}"
    )


def _restart_kernel(nb_runner):
    """Perform a REAL kernel restart and re-establish the notebook path + cash."""
    import asyncio

    nb_runner.restart()
    nb_runner._inject_notebook_path()


def test_append_writer_not_refired_after_restart(nb_runner, tmp_path):
    """CAS-153 round-3 (headline correctness gate): a non-idempotent append-mode
    writer, run once, must NOT re-fire when only a downstream reader runs after a
    kernel restart.

    Post-restart ``executed_write_stmt_codes`` is empty, so the writer looks
    "changed" and the old planner re-scheduled it during freshness simulation —
    appending a SECOND line to the log on disk (a re-fired side effect) and
    handing the reader 2 (truth: 1). The persisted write-provenance +
    output-file freshness check must recognise the effect is already applied and
    leave the writer alone.
    """
    log = _p(tmp_path / "audit.log")
    nb_runner.create_notebook([
        "import cash",                                                    # 1: import
        "%cash_on",                                                       # 2: enable
        f"with open('{log}', 'a') as f:\n    f.write('entry\\n')\nprint('appended')",  # 3: append writer
        f"with open('{log}') as f:\n    nlines = sum(1 for _ in f)\nprint('nlines =', nlines)",  # 4: reader
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "nlines = 1" in nb_runner.get_output(4)
    with open(log) as fh:
        assert sum(1 for _ in fh) == 1

    _restart_kernel(nb_runner)
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)

    # Run ONLY the reader. The writer must not re-fire.
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "nlines = 1" in out, (
        f"reader saw a re-fired append (log grew) after restart: {out!r}"
    )
    with open(log) as fh:
        on_disk = sum(1 for _ in fh)
    assert on_disk == 1, (
        f"append writer re-fired during freshness simulation: audit.log grew to "
        f"{on_disk} lines on disk (a re-run non-idempotent side effect)"
    )


def test_writer_refired_after_restart_when_output_deleted(nb_runner, tmp_path):
    """The freshness short-circuit must NOT suppress a writer whose output file
    is gone: a deleted output makes the recorded snapshot stale, so the writer
    is re-scheduled and re-creates the file.
    """
    out_path = _p(tmp_path / "made.txt")
    nb_runner.create_notebook([
        "import cash",                                                        # 1
        "%cash_on",                                                           # 2
        f"with open('{out_path}', 'w') as f:\n    f.write('payload')\nprint('wrote')",  # 3: writer
        f"with open('{out_path}') as f:\n    body = f.read()\nprint('body =', body)",   # 4: reader
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "body = payload" in nb_runner.get_output(4)

    _restart_kernel(nb_runner)
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)

    # Delete the output on disk: the recorded snapshot is now stale.
    import os
    os.remove(tmp_path / "made.txt")

    # Run ONLY the reader: upstream must re-fire the writer to re-create the file.
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "body = payload" in out, (
        f"deleted output was not re-created: writer wrongly skipped: {out!r}"
    )
    assert os.path.exists(tmp_path / "made.txt"), (
        "writer was not re-fired to re-create its deleted output file"
    )
