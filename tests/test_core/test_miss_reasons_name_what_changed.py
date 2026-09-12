"""Round 20: miss reasons and debug output that told the user the wrong thing.

* Pool calls after `cash clear --function` read "new arguments" (serial calls
  said "entry gone"): the threads read the stored-key record while another
  thread was replacing it, got nothing, and fell back to call-to-call.
* A changed parameter DEFAULT read "new arguments" for all but the first call.
* "code or state changed" never said WHAT changed -- all five testers.
* Changing `file_hash_full_max_bytes` read "content changed" for files whose
  bytes had not moved.
* `explain()` and HIT lines printed a sampled fingerprint like a full hash.
* CASH_DEBUG logged "Cannot attach _cash_lineage_hash to dict" on every call.
* CASH_SUMMARY went through the application's formatter at INFO and raw to
  stderr at WARNING.

Cross-run cases are fresh processes on one cache, reading `CASH_SUMMARY`.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

JOB = textwrap.dedent('''
    import sys, time
    from concurrent.futures import ThreadPoolExecutor
    import cash
    {rank_def}

    TOP_K = {top_k}

    @cash.cache
    def f(x, rounds={rounds}):
        time.sleep(0.2)  # @cash:assume-safe
        return _rank(x) * TOP_K + rounds

    args = [int(a) for a in sys.argv[2:]]
    if sys.argv[1] == "pool":
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(f, args))
    else:
        for a in args:
            f(a)
''')

RANK_HERE = "def _rank(x):\n    return x + 1\n"
RANK_MOVED = "from helpers import _rank\n"


def _write(proj, *, top_k=3, rounds=60, moved=False):
    (proj / "helpers.py").write_text(RANK_HERE, encoding="utf-8")
    rank_def = RANK_MOVED if moved else RANK_HERE
    (proj / "job.py").write_text(
        JOB.format(rank_def=rank_def, top_k=top_k, rounds=rounds), encoding="utf-8")


def _run(proj, mode, args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"),
               CASH_SUMMARY="1")
    p = subprocess.run([sys.executable, "job.py", mode, *map(str, args)],
                       cwd=str(proj), env=env, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stderr


def _missed(summary: str) -> str:
    return next((line.strip() for line in summary.splitlines() if "missed:" in line), "")


def test_pool_misses_after_a_clear_say_the_entries_are_gone(tmp_path):
    _write(tmp_path)
    _run(tmp_path, "pool", range(12))
    for entry in (tmp_path / ".cash").glob("*.entry"):
        entry.unlink()
    summary = _run(tmp_path, "pool", range(12))
    assert _missed(summary) == "missed: 12 entry gone", summary


def test_a_changed_default_is_a_code_change_for_every_call(tmp_path):
    _write(tmp_path, rounds=60)
    _run(tmp_path, "serial", [1, 2, 3, 4])
    _write(tmp_path, rounds=80)
    summary = _run(tmp_path, "serial", [1, 2, 3, 4])
    assert _missed(summary) == "missed: 4 code or state changed", summary
    assert "its own source changed" in summary, summary


def test_an_edited_global_is_named(tmp_path):
    _write(tmp_path, top_k=3)
    _run(tmp_path, "serial", [1])
    _write(tmp_path, top_k=4)
    summary = _run(tmp_path, "serial", [1])
    assert "code or state changed (1x): global TOP_K changed" in summary, summary


def test_a_moved_helper_is_named_as_moved(tmp_path):
    _write(tmp_path, moved=False)
    _run(tmp_path, "serial", [1])
    _write(tmp_path, moved=True)
    summary = _run(tmp_path, "serial", [1])
    assert "moved to" in summary and "_rank" in summary, summary


def test_new_arguments_under_unchanged_code_stay_new_arguments(tmp_path):
    """Control: the earlier-run rule must not fire when the code is the same."""
    _write(tmp_path)
    _run(tmp_path, "serial", [1])
    summary = _run(tmp_path, "serial", [2])
    assert _missed(summary) == "missed: 1 new arguments", summary


# -- in-process ---------------------------------------------------------------

def _stored(path, cap):
    from cash.notebook.file_dep_snapshot import file_content_hash
    st = os.stat(path)
    rec = {"mtime": st.st_mtime, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
           "hash": file_content_hash(str(path), st.st_size, cap)}
    if st.st_size > cap:
        rec.update(ctime=st.st_ctime, ctime_ns=st.st_ctime_ns)
    return rec


@pytest.mark.parametrize("recorded_cap, checked_cap", [(1000, 10**6), (10**6, 1000)])
def test_a_changed_hashing_threshold_is_not_called_a_content_change(
        tmp_path, recorded_cap, checked_cap):
    from cash.notebook.file_dep_snapshot import file_dep_is_fresh
    data = tmp_path / "data.bin"
    data.write_bytes(b"x" * 2000)
    stored = _stored(data, recorded_cap)
    assert file_dep_is_fresh(str(data), stored, full_hash_max=checked_cap) == (False, "hash-mode")


def test_a_real_edit_in_one_regime_is_still_a_content_change(tmp_path):
    from cash.notebook.file_dep_snapshot import file_dep_is_fresh
    data = tmp_path / "data.bin"
    data.write_bytes(b"x" * 2000)
    stored = _stored(data, 10**6)
    data.write_bytes(b"y" * 2000)
    assert file_dep_is_fresh(str(data), stored, full_hash_max=10**6) == (False, "content")


def test_a_sampled_fingerprint_is_labelled_and_a_hit_says_it_trusts_timestamps():
    from cash.core import Cash, _describe_file_deps
    shown = _describe_file_deps({"big.npy": {"size": 3 << 28, "hash": "ab" * 32,
                                             "ctime_ns": 1, "ctime": 1.0}})
    assert "sampled hash" in shown["big.npy"], shown
    line = Cash._describe_call({"func_name": "m.f", "cache_hit": True, "cache_key": "k",
                                "time_saved": 1.0, "execution_time": 0.001,
                                "sampled_files": ("C:/data/big.npy",)})
    assert "trusts the timestamps of big.npy" in line, line


def test_an_untaggable_result_is_not_logged_on_every_call(tmp_path, caplog):
    from cash import Cash
    c = Cash(cache_dir=str(tmp_path / "c"))

    @c.cache
    def rows(n):
        return {"n": n}

    with caplog.at_level(logging.DEBUG, logger="cash"):
        for n in range(4):
            rows(n)
    said = [r.getMessage() for r in caplog.records]
    assert not any("Cannot attach" in m for m in said), said
    assert sum("cannot carry a lineage tag" in m for m in said) <= 1, said


def test_the_summary_goes_through_the_applications_handler_at_any_level(tmp_path, capsys):
    from cash import Cash
    c = Cash(cache_dir=str(tmp_path / "c"))

    @c.cache
    def g(n):
        return n

    g(1)
    root = logging.getLogger()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("LOG %(levelname)s %(name)s: %(message)s"))
    saved_level, saved_handlers = root.level, root.handlers[:]
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    try:
        capsys.readouterr()
        c._print_run_summary()
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
    assert stream.getvalue().startswith("LOG INFO cash.summary: cash:"), stream.getvalue()
    assert "cash:" not in capsys.readouterr().err
