"""A warning is one line by default; ``warning_detail = "full"`` prints all of it.

A seven-line observed-effects warning (three file paths, a 430-character fix
sentence, the link and Python's echo of the calling line) buried a script's
own output. The short form keeps what a reader needs at a glance -- the code,
what happened, where to read more -- and the full text stays where the line
says: ``f.cache_info()['warnings']``.
"""

from __future__ import annotations

import textwrap
import warnings

import pytest

from cash import Cash
from cash.diagnostics import format_diagnostic, short_diagnostic, warn_diagnostic
from cash.exceptions import CashCacheIneffectiveWarning, CashImpurityWarning
from tests._scripts import run_python

URL = "https://cash-lib.readthedocs.io/en/latest/warnings/#cache-thrash"


def test_the_short_form_is_the_first_sentence_and_the_link():
    from cash.diagnostics import short_diagnostic

    text = short_diagnostic("CACHE-THRASH", "the cache is full at its 500 MB cap. Each write evicts another.")
    assert text == f"[CACHE-THRASH] the cache is full at its 500 MB cap. ({URL})"


def test_a_list_collapses_to_a_count_and_its_first_item():
    from cash.diagnostics import short_diagnostic

    what = (
        "@cash.cache on f: the first call wrote files.\n"
        "  in helper (mod.py):\n"
        "    line 3: /home/me/project/__pycache__/x.nbi.tmp.1a\n"
        "    line 4: /home/me/project/__pycache__/x.nbc.tmp.2b\n"
    )
    text = short_diagnostic("CACHE-THRASH", what, more="f.cache_info()['warnings']")
    assert "\n" not in text
    assert text == (
        "[CACHE-THRASH] @cash.cache on f: the first call wrote files: 2 findings, "
        f"e.g. line 3: .../__pycache__/x.nbi.tmp.1a (f.cache_info()['warnings']; {URL})"
    )


def _emitted(**config):
    cash = Cash(backend="memory", register_magic=False, **config)
    from cash import _active

    token = _active.ACTIVE_CONFIG.set(cash.config)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "CACHE-THRASH",
                "the cache is full at its 500 MB cap. It evicts.",
                "raise it.",
            )
    finally:
        _active.ACTIVE_CONFIG.reset(token)
    [w] = caught
    return str(w.message)


def test_a_warning_is_one_line_by_default():
    assert _emitted() == f"[CACHE-THRASH] the cache is full at its 500 MB cap. Fix: raise it. ({URL})"


def test_the_one_line_keeps_the_whole_fix():
    # The way out (an annotation, "re-run cell 3") is what a reader acts on.
    text = short_diagnostic(
        "CACHE-THRASH",
        "the cache is full.",
        "split it. If that is not\n  possible, put `# @cash:no-cache` above it.",
    )
    assert (
        text
        == f"[CACHE-THRASH] the cache is full. Fix: split it. If that is not possible, put `# @cash:no-cache` above it. ({URL})"
    )


def test_a_first_sentence_that_says_only_what_takes_the_next_one():
    # "result not cached." alone would leave out why.
    assert short_diagnostic("CACHE-THRASH", "@cash.cache on f: result not cached. A Figure is live.") == (
        f"[CACHE-THRASH] @cash.cache on f: result not cached. A Figure is live. ({URL})"
    )


def test_a_first_sentence_without_a_full_stop_gets_one_before_the_fix():
    # NOTEBOOK-UNSEEN-WRITES read "... would not change it Fix: name ...".
    assert short_diagnostic("CACHE-THRASH", "the cache is full", "raise it.") == (
        f"[CACHE-THRASH] the cache is full. Fix: raise it. ({URL})"
    )


def test_inline_code_prints_with_one_backtick():
    text = short_diagnostic("CACHE-THRASH", "the cache is full: ``max_cache_size`` is 500 MB.", "raise ``x``.")
    assert "``" not in text
    assert "`max_cache_size`" in text and "`x`" in text


def test_a_handed_state_changing_function_says_the_call_runs_uncached(cash_instance):
    """The first sentence of IMPURE-HANDED-CALLABLE carries the consequence:
    a longer one was cut before it, and the line said only what was found."""
    count = []

    def bump():
        count.append(1)

    @cash_instance.cache
    def run(fn):
        fn()
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(bump)
    [line] = [str(w.message) for w in caught if getattr(w.message, "code", None) == "IMPURE-HANDED-CALLABLE"]
    assert "so calls like this run uncached" in line, line
    assert "``" not in line, line


def test_full_detail_prints_the_full_text_unchanged():
    expected = format_diagnostic("CACHE-THRASH", "the cache is full at its 500 MB cap. It evicts.", "raise it.")
    assert _emitted(warning_detail="full") == expected
    assert expected == f"[CACHE-THRASH] the cache is full at its 500 MB cap. It evicts.\n  Fix: raise it.\n  {URL}"


def test_an_unknown_detail_is_refused():
    assert Cash(backend="memory", register_magic=False, warning_detail="full").config.warning_detail == "full"
    with pytest.raises(ValueError, match="warning_detail"):
        Cash(backend="memory", register_magic=False, warning_detail="long")


def _writes_a_zip(tmp_path):
    import zipfile

    def plain(name):
        with zipfile.ZipFile(str(tmp_path / name), "w") as z:
            names = z.namelist()
        return str(names)

    return plain


def _observed(cash, *funcs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i, f in enumerate(funcs):
            f(f"{i}.zip")
    return [str(w.message) for w in caught if getattr(w.message, "code", None) == "IMPURE-OBSERVED-EFFECTS"]


def test_the_function_log_keeps_the_full_text(tmp_path):
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    archive = cash.cache(_writes_a_zip(tmp_path))
    [shown] = _observed(cash, archive)
    assert "\n" not in shown
    assert "file write: .../" in shown, shown
    assert "plain.cache_info()['warnings']" in shown
    assert "Fix: split the function" in shown
    assert "`# @cash:assume-safe`" in shown, "the one line keeps the way out"
    [logged] = [w["message"] for w in archive.cache_info()["warnings"] if w["code"] == "IMPURE-OBSERVED-EFFECTS"]
    assert "\n  Fix: split the function" in logged
    assert str(tmp_path) in logged, "the log lists the full path"


def test_the_same_code_on_a_second_function_keeps_its_finding(tmp_path):
    """The explanation and the fix are said once; what the second function
    does is its own and is said again."""
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    first = cash.cache(_writes_a_zip(tmp_path))
    second_body = _writes_a_zip(tmp_path)
    second_body.__name__ = second_body.__qualname__ = "other"
    second = cash.cache(second_body)
    shown = _observed(cash, first, second)
    assert len(shown) == 2, shown
    assert shown[0].startswith("[IMPURE-OBSERVED-EFFECTS] @cash.cache on ")
    assert shown[1].startswith("[IMPURE-OBSERVED-EFFECTS] other: "), shown[1]
    assert "file write: .../" in shown[1] and "1.zip" in shown[1], shown[1]
    assert shown[1].endswith(". Same fix as for plain. (other.cache_info()['warnings'])"), shown[1]
    assert "Fix:" not in shown[1] and "\n" not in shown[1]
    assert "Fix:" in second.cache_info()["warnings"][-1]["message"], "the log still has it all"


SIDE_EFFECTS = """
import subprocess, sys, warnings
import cash

@cash.cache
def report(x):
    print("working on", x)
    return x

@cash.cache
def export(x):
    with open("out.txt", "w") as f:
        f.write(str(x))
    return x

@cash.cache
def notify(x):
    subprocess.run([sys.executable, "-c", "pass"])
    return x

with warnings.catch_warnings(record=True) as got:
    warnings.simplefilter("always")
    report(1); export(1); notify(1)
for w in got:
    print("W", str(w.message))
"""


def test_each_function_with_a_side_effect_names_its_own(tmp_path):
    """Three functions under one code: a ``print``, a file write and a
    subprocess. The second and third lines name the write and the
    subprocess, which a hit skips; only the fix is said once."""
    (tmp_path / "main.py").write_text(SIDE_EFFECTS, encoding="utf-8")
    out = run_python("main.py", cwd=tmp_path).stdout
    lines = [line[2:] for line in out.splitlines() if line.startswith("W [IMPURE-SIDE-EFFECTS]")]
    assert len(lines) == 3, out
    assert "@cash.cache on" in lines[0] and "print()" in lines[0] and "Fix:" in lines[0]
    assert lines[1].startswith("[IMPURE-SIDE-EFFECTS] export: 2 findings, e.g. line 12: [impure_call] open()"), lines[1]
    assert lines[2].startswith("[IMPURE-SIDE-EFFECTS] notify: line 18: [impure_call] subprocess.run()"), lines[2]
    for line, name in ((lines[1], "export"), (lines[2], "notify")):
        assert line.endswith(f". Same fix as for report. ({name}.cache_info()['warnings'])"), line


def test_a_repeat_without_a_list_keeps_its_first_sentence():
    from cash.diagnostics import repeat_diagnostic

    text = repeat_diagnostic(
        "CACHE-THRASH",
        "draw",
        "@cash.cache on pkg.draw: Unseeded randomness detected: ``random.random()`` at line 4. A seed set "
        "before the call is part of the key.",
        first="sample",
        more="draw.cache_info()['warnings']",
    )
    assert text == (
        "[CACHE-THRASH] draw: Unseeded randomness detected: `random.random()` at line 4. "
        "Same fix as for sample. (draw.cache_info()['warnings'])"
    )
    short = repeat_diagnostic(
        "CACHE-THRASH", "f", "@cash.cache on f: result not cached. A Figure is live.", first="g", more="m"
    )
    assert short == "[CACHE-THRASH] f: result not cached. A Figure is live. Same fix as for g. (m)"


def test_full_detail_repeats_the_full_text_per_function(tmp_path):
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, warning_detail="full")
    first = cash.cache(_writes_a_zip(tmp_path))
    second_body = _writes_a_zip(tmp_path)
    second_body.__name__ = second_body.__qualname__ = "other"
    shown = _observed(cash, first, cash.cache(second_body))
    assert len(shown) == 2 and all("\n  Fix: " in m for m in shown), shown


SCRIPT = """
import warnings, zipfile
import cash

@cash.cache
def plain(name):
    with zipfile.ZipFile(name, "w") as z:
        names = z.namelist()
    return str(names)

print("RESULT", plain("a.zip"))  # the echoed line
warnings.warn("someone else's")  # their echoed line
"""


def _run(tmp_path, *args, env=None, check=True):
    """Run SCRIPT on a cache of its own: a hit would not warn."""
    run = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
    run.mkdir()
    (run / "main.py").write_text(textwrap.dedent(SCRIPT), encoding="utf-8")
    return run_python(*args, "main.py", cwd=run, env=env, check=check)


def test_a_script_shows_one_line_and_no_echoed_source(tmp_path):
    err = _run(tmp_path).stderr
    lines = [line for line in err.splitlines() if line.strip()]
    ours = [line for line in lines if "IMPURE-OBSERVED-EFFECTS" in line]
    assert len(ours) == 1, err
    assert "main.py:11: CashImpurityWarning: [IMPURE-OBSERVED-EFFECTS]" in ours[0]
    assert "# the echoed line" not in err, err
    assert "# their echoed line" in err, "another warning keeps Python's own format"
    assert len(lines) == 3, err


def test_the_variable_asks_for_the_full_text(tmp_path):
    err = _run(tmp_path, env={"CASH_WARNING_DETAIL": "full"}).stderr
    assert "\n  Fix: split the function" in err
    assert "# the echoed line" not in err


def test_filters_still_act_on_the_category(tmp_path):
    err = _run(tmp_path, "-W", "ignore::cash.CashImpurityWarning").stderr
    assert "IMPURE-OBSERVED-EFFECTS" not in err
    raised = _run(tmp_path, "-W", "error::cash.CashImpurityWarning", check=False)
    assert raised.returncode != 0
    assert "CashImpurityWarning: [IMPURE-OBSERVED-EFFECTS]" in raised.stderr


def test_a_category_filter_in_code_still_silences_it(tmp_path):
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    archive = cash.cache(_writes_a_zip(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=CashImpurityWarning)
        archive("a.zip")
    assert not [w for w in caught if issubclass(w.category, CashImpurityWarning)]
