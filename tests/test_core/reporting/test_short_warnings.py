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
from cash.diagnostics import format_diagnostic, warn_diagnostic
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
            warn_diagnostic(CashCacheIneffectiveWarning, "CACHE-THRASH", "the cache is full. It evicts.", "raise it.")
    finally:
        _active.ACTIVE_CONFIG.reset(token)
    [w] = caught
    return str(w.message)


def test_a_warning_is_one_line_by_default():
    assert _emitted() == f"[CACHE-THRASH] the cache is full. ({URL})"


def test_full_detail_prints_the_full_text_unchanged():
    expected = format_diagnostic("CACHE-THRASH", "the cache is full. It evicts.", "raise it.")
    assert _emitted(warning_detail="full") == expected
    assert expected == f"[CACHE-THRASH] the cache is full. It evicts.\n  Fix: raise it.\n  {URL}"


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
    assert "Fix:" not in shown
    [logged] = [w["message"] for w in archive.cache_info()["warnings"] if w["code"] == "IMPURE-OBSERVED-EFFECTS"]
    assert "\n  Fix: split the function" in logged
    assert str(tmp_path) in logged, "the log lists the full path"


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
    (run / "main.py").write_text(textwrap.dedent(SCRIPT))
    return run_python(*args, "main.py", cwd=run, env=env, check=check)


def test_the_variable_asks_for_the_full_text(tmp_path):
    err = _run(tmp_path, env={"CASH_WARNING_DETAIL": "full"}).stderr
    assert "\n  Fix: split the function" in err


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
