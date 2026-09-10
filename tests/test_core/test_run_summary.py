"""An end-of-run summary, for the script user who cannot see the badge.

A notebook says per statement whether it ran or restored. A script said
nothing, so a user who wanted to know which parts of his run recomputed added
``print("X done")`` to each branch by hand. ``show_stats()`` existed and he
never found it -- and would not have been helped if he had, because it printed
a backend name and a function count rather than hits and misses.

Two things are pinned here. The summary itself, off by default because a
library that prints uninvited is a library people filter. And the fact that
``show_stats()`` in a script prints the same table: its documented "prints a
text summary instead" fallback sat behind ``except (ImportError, RuntimeError)``
while the dashboard *printed* "ipywidgets is required" and returned normally,
so the fallback was unreachable and the documented behaviour never happened.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import cash


def _cash(tmp_path, **kwargs):
    return cash.Cash(cache_dir=str(tmp_path / "cache"), **kwargs)


def _exercise(c):
    @c.cache(assume_safe=True)
    def work(n):
        return n + 1

    work(1)
    work(1)      # hit
    work(2)      # miss
    return work


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_nothing_called_produces_no_table(tmp_path):
    """Empty, not a header over an empty list -- so callers can print blindly."""
    assert _cash(tmp_path).run_summary() == ""


def test_it_reports_hits_and_misses_per_function(tmp_path):
    c = _cash(tmp_path)
    _exercise(c)
    text = c.run_summary()
    assert "work" in text
    assert "1 hit," in text
    assert "2 misses" in text
    assert "1 of 3 calls restored" in text


def test_one_is_singular_and_the_comma_stays_put(tmp_path):
    """Padding the WORD rather than the token produced "1 hit ,"."""
    c = _cash(tmp_path)
    _exercise(c)
    text = c.run_summary()
    assert "1 hit," in text
    assert "1 hit ," not in text
    assert "1 hits" not in text


def test_functions_are_ranked_by_time_saved(tmp_path):
    """The one that saved you the most is the one you want to see first."""
    c = _cash(tmp_path)

    @c.cache(assume_safe=True)
    def cheap(n):
        return n

    @c.cache(assume_safe=True)
    def dear(n):
        return n

    cheap(1)
    dear(1)
    c._function_stats["__main__.dear" if "__main__.dear" in c._function_stats
                      else next(k for k in c._function_stats if "dear" in k)
                      ]["total_time_saved"] = 99.0
    text = c.run_summary()
    assert text.index("dear") < text.index("cheap")


def test_the_table_is_ascii(tmp_path):
    """It lands on a cp1252 Windows console more often than not."""
    c = _cash(tmp_path)
    _exercise(c)
    assert c.run_summary().isascii()


# ---------------------------------------------------------------------------
# How it is switched on
# ---------------------------------------------------------------------------


def test_off_by_default(tmp_path):
    c = _cash(tmp_path)
    assert c.config.summary is False


def test_the_constructor_switches_it_on(tmp_path):
    assert _cash(tmp_path, summary=True).config.summary is True


def test_the_env_var_switches_it_on_with_no_code_change(tmp_path):
    """The spelling that matters: it needs no edit to the script you are running.

    One config field yields the constructor, ``cash.configure``, this, and a
    TOML key, because the config layer maps every field to all four.
    """
    script = tmp_path / "s.py"
    script.write_text(textwrap.dedent("""
        import cash
        c = cash.Cash(cache_dir="cache")

        @c.cache(assume_safe=True)
        def work(n):
            return n + 1

        work(1)
        work(1)
    """), encoding="utf-8")

    def run(env_value):
        env = dict(os.environ)
        env.pop("CASH_SUMMARY", None)
        if env_value is not None:
            env["CASH_SUMMARY"] = env_value
        return subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, cwd=str(tmp_path), env=env,
                              encoding="utf-8", errors="replace")

    assert "calls restored" not in run(None).stderr, "printed without being asked"
    shown = run("1")
    # stderr since CAS-120: stdout is the program's own output.
    assert "calls restored" in shown.stderr
    assert "calls restored" not in shown.stdout


def test_the_summary_never_breaks_a_finished_run(tmp_path, monkeypatch):
    """It runs during interpreter shutdown; a traceback there helps nobody."""
    c = _cash(tmp_path)
    _exercise(c)
    monkeypatch.setattr(type(c), "run_summary",
                        lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    c._print_run_summary()      # must not raise


# ---------------------------------------------------------------------------
# show_stats() in a script
# ---------------------------------------------------------------------------


def test_show_stats_in_a_script_prints_the_table(tmp_path, capsys):
    c = _cash(tmp_path)
    _exercise(c)
    c.show_stats()
    out = capsys.readouterr().out
    assert "calls restored" in out
    assert "ipywidgets" not in out, (
        "the documented script fallback is unreachable again: the dashboard "
        "prints its own failure and returns, so catching ImportError never fires"
    )


def test_show_stats_says_so_when_there_is_nothing_yet(tmp_path, capsys):
    _cash(tmp_path).show_stats()
    out = capsys.readouterr().out
    assert "no cached function" in out
    assert "ipywidgets" not in out


# ---------------------------------------------------------------------------
# Which cache it ran against
# ---------------------------------------------------------------------------

def test_the_summary_names_the_cache_directory(tmp_path):
    """The one line that answers "why is nothing cached".

    Every version of that question -- a scheduled job whose cwd-relative cache
    is somewhere else, a path typed with one backslash too few, a container
    volume that is not the one you meant -- is answered or excluded by seeing
    the directory. A round-15 tester spent an arm of their round on a cache
    directory that was not the one they thought they had set, with
    `CASH_SUMMARY=1` on the whole time and nothing in its output to say so.
    """
    c = _cash(tmp_path)
    _exercise(c)
    assert str(tmp_path / "cache") in c.run_summary()


def test_naming_it_does_not_build_a_backend(tmp_path):
    """The summary runs from an atexit handler; it must not create anything.

    A cache directory conjured into existence by the summary would be a new
    defect of exactly the kind the line is there to diagnose.
    """
    c = _cash(tmp_path)
    assert c._backend is None, "the fixture must start with the backend unbuilt"

    c.run_summary()

    assert c._backend is None
    assert not (tmp_path / "cache").exists()
