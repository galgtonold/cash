"""A script user can find out why a call recomputed.

CAS-120, round 17: four of five testers could not.

- ``CASH_SUMMARY=1`` printed into stdout -- a CLI's report, a pipe, a JSON
  response -- and said "1 miss" and nothing else. The miss was the 0.1 s
  persistence floor: the value never reached disk, so every new process
  recomputed it.
- ``CASH_DEBUG=1`` printed nothing at all in a script, because the flag set an
  attribute and a script has no logging configured.
- ``explain()`` said "first call, or the cache was cleared" on the fourth
  identical call, never said which part of the key moved, showed a cache key
  where ``cash clear --entry`` wants an entry id, and could not list the files
  an entry depends on.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
import time

import pytest

from cash import Cash

pytestmark = pytest.mark.core


def _run(tmp_path, body, **env_extra):
    script = tmp_path / "job.py"
    script.write_text(body, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / ".cash")
    env.update(env_extra)
    return subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, cwd=str(tmp_path), env=env,
                          encoding="utf-8", errors="replace")


_FAST_JOB = textwrap.dedent("""
    import cash

    @cash.cache(assume_safe=True)
    def work(n):
        return n + 1

    print("REPORT", work(1), work(1), work(2))
""")


# -- where it prints ---------------------------------------------------------

def test_the_summary_goes_to_stderr_not_into_the_programs_output(tmp_path):
    out = _run(tmp_path, _FAST_JOB, CASH_SUMMARY="1")
    assert out.stdout.strip() == "REPORT 2 2 3", "the program's output was polluted"
    assert "calls restored" in out.stderr


def test_the_summary_survives_a_failing_exit(tmp_path):
    """sys.exit(1) and an uncaught exception both still report."""
    for ending in ("raise SystemExit(1)", "raise RuntimeError('boom')"):
        out = _run(tmp_path, _FAST_JOB + ending + "\n", CASH_SUMMARY="1")
        assert out.returncode != 0
        assert "calls restored" in out.stderr, ending


def test_cash_debug_prints_each_decision_in_a_plain_script(tmp_path):
    """THE SILENCE: CASH_DEBUG=1 printed nothing without logging configured."""
    out = _run(tmp_path, _FAST_JOB, CASH_DEBUG="1")
    assert out.stdout.strip() == "REPORT 2 2 3"
    lines = [ln for ln in out.stderr.splitlines() if ln.startswith("cash.calls:")]
    assert len(lines) == 3, out.stderr
    assert "MISS" in lines[0] and "no entry yet" in lines[0]
    assert "HIT" in lines[1]
    assert "MISS" in lines[2] and "new arguments" in lines[2]
    assert "kept in RAM only" in lines[0], "the floor went unexplained"


def test_cash_debug_uses_the_applications_logging_when_there_is_some(tmp_path):
    """No second handler: the app's format and destination win."""
    body = textwrap.dedent("""
        import logging, sys
        logging.basicConfig(stream=sys.stdout, format="APP %(name)s %(message)s")
    """) + _FAST_JOB
    out = _run(tmp_path, body, CASH_DEBUG="1")
    assert "APP cash.calls MISS" in out.stdout
    assert "cash.calls:" not in out.stderr, "a duplicate handler was added"


def test_configure_debug_turns_it_on_too(tmp_path):
    """The runtime switch set the flag and nothing that reads it."""
    body = "import cash\ncash.configure(debug=True)\n" + _FAST_JOB
    out = _run(tmp_path, body)
    assert "cash.calls: MISS" in out.stderr, out.stderr


def test_no_per_call_lines_without_being_asked(tmp_path):
    out = _run(tmp_path, _FAST_JOB)
    assert "cash.calls" not in out.stderr + out.stdout


# -- what it says ------------------------------------------------------------

@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def test_each_reason_is_named(c, tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("abc", encoding="utf-8")
    g = {"FACTOR": 2}

    @c.cache(assume_safe=True)
    def length(path):
        with open(path, encoding="utf-8") as f:
            return len(f.read()) * g["FACTOR"]

    length(str(data))                        # no entry yet
    length(str(data))                        # hit
    data.write_text("abcdef", encoding="utf-8")
    length(str(data))                        # file changed
    length(str(data) + "")                   # hit
    other = tmp_path / "other.txt"
    other.write_text("x", encoding="utf-8")
    length(str(other))                       # new arguments

    assert length.cache_info()["miss_reasons"] == {
        "no entry yet": 1, "file changed": 1, "new arguments": 1}


def test_a_changed_global_is_a_state_change(c, tmp_path, monkeypatch):
    """A real file: a function with no retrievable source has no globals to fold."""
    (tmp_path / "why_mod.py").write_text(textwrap.dedent("""
        SCALE = 2

        def scaled(x):
            return x * SCALE
    """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import why_mod
    try:
        scaled = c.cache(assume_safe=True)(why_mod.scaled)
        scaled(3)
        why_mod.SCALE = 5
        assert scaled(3) == 15
        assert scaled.cache_info()["miss_reasons"].get("code or state changed") == 1
    finally:
        sys.modules.pop("why_mod", None)


def test_a_result_that_was_not_stored_says_why_next_time(c):
    @c.cache(assume_safe=True, cache_if=lambda r: r > 0)
    def maybe(x):
        return x

    maybe(0)
    maybe(0)
    assert maybe.cache_info()["miss_reasons"]["not stored last time"] == 1
    assert "cache_if returned False" in maybe.explain(0).details["why"]


def test_a_ttl_expiry_is_a_ttl_expiry(c):
    @c.cache(assume_safe=True, ttl=1)
    def short(x):
        return x

    short(1)
    time.sleep(1.2)
    assert short.explain(1).reason == "ttl_expired"
    short(1)
    assert short.cache_info()["miss_reasons"]["ttl expired"] == 1


def test_the_summary_says_what_stayed_in_ram(c):
    @c.cache(assume_safe=True)
    def fast(n):
        return n

    @c.cache(assume_safe=True)
    def slow(n):
        time.sleep(0.25)
        return n

    fast(1)
    slow(1)
    blocks = _summary_blocks(c.run_summary())
    fast_notes = next(v for k, v in blocks.items() if k.endswith(".fast"))
    slow_notes = next(v for k, v in blocks.items() if k.endswith(".slow"))
    assert "kept in RAM only" in fast_notes and "persistence floor" in fast_notes
    assert "RAM only" not in slow_notes, "a persisted result was reported as RAM-only"


def test_cash_s_own_work_does_not_push_a_trivial_function_past_the_floor(c, monkeypatch):
    """The floor is judged on the body's time. It was the call's wall-clock
    time, cash's own key work included -- which the next process pays again
    whether the entry exists or not -- so on a busy Windows runner a function
    that returns at once was persisted, and this summary test failed. Here
    the key work is made slow on purpose."""
    real = type(c)._serialize_args

    def slow_key(self, *args, **kwargs):
        time.sleep(0.15)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(c), "_serialize_args", slow_key)

    @c.cache(assume_safe=True)
    def fast(n):
        return n

    fast(1)
    notes = next(v for k, v in _summary_blocks(c.run_summary()).items() if k.endswith(".fast"))
    assert "kept in RAM only" in notes and "persistence floor" in notes, notes


def _summary_blocks(text: str) -> dict[str, str]:
    """``{function: its indented note lines}`` from a run summary."""
    blocks: dict[str, str] = {}
    current = None
    for line in text.splitlines()[1:]:
        if line.startswith("      ") and current:
            blocks[current] += line.strip() + "\n"
        elif line.startswith("  ") and not line.startswith("  cache:"):
            current = line.split()[0]
            blocks[current] = ""
    return blocks


# -- explain() and inspect ---------------------------------------------------

def test_explain_names_the_moved_part_of_the_key(c):
    @c.cache(assume_safe=True)
    def f(x):
        return x

    f(1)
    why = f.explain(2).details["why"]
    assert why.startswith("new arguments"), why


def test_explain_gives_the_entry_id_cash_clear_takes(c, tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("abc", encoding="utf-8")

    @c.cache(assume_safe=True)
    def read(path):
        time.sleep(0.2)                      # past the floor, so it reaches disk
        with open(path, encoding="utf-8") as f:
            return f.read()

    read(str(data))
    e = read.explain(str(data))
    assert e.entry_id == hashlib.sha256(e.cache_key.encode()).hexdigest()[:12]
    assert f"entry_id: {e.entry_id}" in str(e)
    assert any(str(data.name) in p for p in e.details["file_deps"]), e.details

    c.shutdown()
    listing = subprocess.run(
        [sys.executable, "-m", "cash", "inspect", str(tmp_path / ".cash"),
         "--function", "read"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert e.entry_id in listing.stdout, listing.stdout + listing.stderr
    assert "reads:" in listing.stdout and data.name in listing.stdout
