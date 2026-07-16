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
    loop = asyncio.get_event_loop()
    loop.run_until_complete(nb_runner.client.km._async_restart_kernel(now=True))
    loop.run_until_complete(nb_runner.client.kc._async_wait_for_ready(timeout=30))
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
