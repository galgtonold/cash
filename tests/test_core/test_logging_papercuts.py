"""Round-18 logging and observability papercuts, one test each.

The "why did that recompute?" tools failed in exactly the environments people
run: pytest (CASH_DEBUG printed nothing, even with -s), a service with its own
dictConfig (cash overrode the level it set; the summary never reached its
log), a crashed run ("5 of 5 calls restored"), a worker pool (summaries
interleaved mid-line), and `cash inspect` (two headers; nothing said an entry
had expired).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
import time

import pytest

import cash
from cash import Cash, FileBackend

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


def _env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONDONTWRITEBYTECODE="1", **extra)
    return env


def test_cash_debug_prints_under_pytest(tmp_path):
    """pytest's logging plugin puts capture handlers on the root logger, and
    cash read that as "the application configured logging"."""
    (tmp_path / "test_job.py").write_text(textwrap.dedent('''
        import cash

        @cash.cache
        def f(x):
            return x + 1

        def test_it():
            assert f(1) == 2
    '''), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "pytest", "-s", "-q", "-p", "no:cacheprovider",
                        "test_job.py"], cwd=str(tmp_path), capture_output=True, text=True,
                       env=_env(tmp_path, CASH_DEBUG="1"), timeout=120)
    assert "1 passed" in p.stdout, p.stdout + p.stderr
    assert "cash.calls: MISS" in p.stdout + p.stderr


def test_an_application_level_on_the_cash_logger_is_kept(tmp_path):
    """dictConfig set `cash` to INFO; CASH_DEBUG lowered it to DEBUG under the
    application on every Cash()."""
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent('''
        import logging, logging.config
        logging.config.dictConfig({"version": 1, "handlers": {"h": {"class": "logging.StreamHandler"}},
                                   "loggers": {"cash": {"level": "INFO", "handlers": ["h"]}}})
        import cash
        cash.Cash(debug=True)
        cash.Cash(debug=True)
        print(logging.getLevelName(logging.getLogger("cash").level))
    '''), encoding="utf-8")
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=_env(tmp_path), timeout=120)
    assert p.stdout.strip().splitlines()[-1] == "INFO", p.stdout + p.stderr


def test_verbose_can_be_configured_and_lines_carry_the_entry_id(tmp_path, caplog):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / ".cash")), register_magic=False)
    c.config.verbose = True

    @c.cache
    def f(x):
        return x + 1

    with caplog.at_level(logging.INFO, logger="cash.calls"):
        f(1)
        f(1)
    lines = [r.getMessage() for r in caplog.records if r.name == "cash.calls"]
    assert any(line.startswith("MISS") for line in lines), lines
    key = f.explain(1).cache_key
    assert any(cash.core.entry_id_of(key) in line for line in lines)


def test_a_call_that_raises_is_logged_and_counted(tmp_path, caplog):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / ".cash")), register_magic=False,
             verbose=True)

    @c.cache
    def boom(x):
        raise ValueError("bad input")

    with caplog.at_level(logging.INFO, logger="cash.calls"), pytest.raises(ValueError):
        boom(1)
    lines = [r.getMessage() for r in caplog.records if r.name == "cash.calls"]
    assert any(line.startswith("RAISE") and "ValueError: bad input" in line for line in lines), lines
    info = boom.cache_info()
    assert info["misses"] == 1 and info["miss_reasons"] == {"raised": 1}
    assert "0 of 1 calls restored" in c.run_summary()


def test_the_summary_reaches_an_application_log_in_one_write(tmp_path, monkeypatch, capsys):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / ".cash")), register_magic=False)

    @c.cache
    def f(x):
        return x

    f(1)
    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Grab()
    logging.getLogger("cash").addHandler(handler)
    monkeypatch.setattr(logging.getLogger("cash"), "level", logging.INFO)
    monkeypatch.setattr("cash.backends._base._in_multiprocessing_child", lambda: True)
    try:
        c._print_run_summary()
    finally:
        logging.getLogger("cash").removeHandler(handler)
    err = capsys.readouterr().err
    # Into the application's log, pid-labelled, and ONLY there: written to
    # stderr as well, with cash's own handler passing it on too, it printed
    # three times (round 19).
    assert records and records[0].startswith(f"cash (pid {os.getpid()}):"), records
    assert "calls restored" not in err, err


def test_a_file_read_by_two_spellings_is_listed_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv").write_text("a\n1\n", encoding="utf-8")
    rec = {"size": 4, "hash": "abc"}
    listed = cash.core._describe_file_deps({"data.csv": rec, str(tmp_path / "data.csv"): rec})
    assert len(listed) == 1


def test_inspect_prints_one_header_and_when_an_entry_expires(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent('''
        import time, cash
        @cash.cache(ttl=3600)
        def f(x):
            time.sleep(0.15)  # @cash:assume-safe
            return x
        f(1)
    '''), encoding="utf-8")
    env = _env(tmp_path)
    subprocess.run([sys.executable, str(script)], env=env, check=True, timeout=120)
    p = subprocess.run([sys.executable, "-m", "cash", "inspect", "--function", "f"],
                       cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
    out = p.stdout
    assert out.count("Cache dir") == 1, out
    assert "EXPIRES" in out and ("in 59m" in out or "in 1h" in out), out


def test_a_summary_the_apps_log_would_drop_goes_to_stderr(tmp_path, monkeypatch, capsys):
    """The app logs, but the `cash` logger is at WARNING: routed to the log,
    the INFO summary would reach no handler and be lost."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / ".cash")), register_magic=False)

    @c.cache
    def f(x):
        return x

    f(1)
    handler = logging.StreamHandler()
    logging.getLogger("cash").addHandler(handler)
    monkeypatch.setattr(logging.getLogger("cash"), "level", logging.WARNING)
    try:
        c._print_run_summary()
    finally:
        logging.getLogger("cash").removeHandler(handler)
    assert "calls restored" in capsys.readouterr().err
