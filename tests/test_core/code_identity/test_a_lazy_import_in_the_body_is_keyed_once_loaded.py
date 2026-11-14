"""A cached function whose body imports a module lazily is keyed once it is loaded.

Reported against nilearn's commit walk: ``from nilearn.regions.signal_extraction
import ...`` inside a function, of a module not loaded yet whose top level
cash cannot read without running it, was said to refuse every call of the
process, not only the first.

The call that first imports such a module runs uncached: its top level may
compute a value from outside its source (the environment, a file), which
neither its file nor a trace of the call would see. Once the module is
loaded the walk reads it, so the calls after it are keyed and served, in this
process and the next, and an edit to the module recomputes, also when only a
later call reaches the import.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

# Not readable without running it: a call cash does not know at the top level.
SIG = textwrap.dedent("""
    import json
    import os

    FACTOR = json.loads("{FACTOR}")
    OFFSET = int(os.environ.get("SIG_OFFSET", "0"))


    def scale(xs):
        return [FACTOR * x for x in xs]
""")

MASKER = textwrap.dedent("""
    import sys
    import cash


    @cash.cache
    def transform(xs, use=True):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        if not use:
            return list(xs)
        from lazypkg.sig import OFFSET, scale

        return [v + OFFSET for v in scale(xs)]
""")

MAIN = textwrap.dedent("""
    import sys
    from lazypkg.masker import transform

    for use in sys.argv[1:]:
        print(transform([1, 2], use == "1"))
""")

UNREAD = "missed: 1 imports a module cash cannot read before it is loaded"


def _write(tmp_path, factor=3):
    pkg = tmp_path / "lazypkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sig.py").write_text(SIG.replace("{FACTOR}", str(factor)), encoding="utf-8")
    (pkg / "masker.py").write_text(MASKER, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")


def _run(tmp_path, *uses, trace="1", offset=None):
    env = {"CASH_TRACE": trace, "CASH_SUMMARY": "1", "SIG_OFFSET": offset}
    proc = run_python("main.py", *uses, cwd=tmp_path, env=env)
    return proc.stdout.split(), proc.stderr


def _runs(err):
    return err.count("[RUN]")


@pytest.mark.parametrize("trace", ["1", "0"])
def test_the_calls_after_the_import_hit(tmp_path, trace):
    """Call 1 imports the module and runs uncached; call 2 is the first
    keyed, and every call after it hits."""
    _write(tmp_path)
    out, err = _run(tmp_path, "1", "1", "1", "1", trace=trace)
    assert out == ["[3,", "6]"] * 4
    assert _runs(err) == 2, err
    assert UNREAD in err and "2 hits" in err, err


@pytest.mark.parametrize("trace", ["1", "0"])
def test_a_second_process_runs_only_its_first_call(tmp_path, trace):
    _write(tmp_path)
    _run(tmp_path, "1", "1", "1", trace=trace)
    out, err = _run(tmp_path, "1", "1", "1", trace=trace)
    assert out == ["[3,", "6]"] * 3
    assert _runs(err) == 1, err
    assert UNREAD in err and "2 hits" in err, err


@pytest.mark.parametrize("trace", ["1", "0"])
def test_an_edit_to_the_lazily_imported_module_recomputes(tmp_path, trace):
    _write(tmp_path)
    _run(tmp_path, "1", "1", "1", trace=trace)
    _run(tmp_path, "1", "1", "1", trace=trace)
    _write(tmp_path, factor=5)
    out, err = _run(tmp_path, "1", "1", "1", trace=trace)
    assert out == ["[5,", "10]"] * 3, f"served the result from before the edit:\n{err}"
    _write(tmp_path, factor=7)
    out, err = _run(tmp_path, "1", "1", trace=trace)
    assert out == ["[7,", "14]"] * 2, f"served the result from before the edit:\n{err}"


@pytest.mark.parametrize("trace", ["1", "0"])
def test_an_edit_reached_only_by_a_later_call_recomputes(tmp_path, trace):
    """Call 1 does not reach the import; a later call does, and must see an
    edit made since that later call was stored."""
    _write(tmp_path)
    _run(tmp_path, "0", "1", "1", trace=trace)
    _run(tmp_path, "1", "0", "1", trace=trace)
    _write(tmp_path, factor=5)
    out, err = _run(tmp_path, "0", "1", "1", trace=trace)
    assert out == ["[1,", "2]", "[5,", "10]", "[5,", "10]"], err
    _write(tmp_path, factor=7)
    out, err = _run(tmp_path, "1", "0", "1", trace=trace)
    assert out == ["[7,", "14]", "[1,", "2]", "[7,", "14]"], err


@pytest.mark.parametrize("trace", ["1", "0"])
def test_a_value_the_top_level_reads_from_the_environment_is_not_served_stale(tmp_path, trace):
    """The file is the same and the call that imports the module ran the same
    code, but the value it bound came from the environment: the first call
    of a process is not served from what an earlier process stored."""
    _write(tmp_path)
    _run(tmp_path, "1", "1", trace=trace, offset="0")
    _run(tmp_path, "1", "1", trace=trace, offset="0")
    out, err = _run(tmp_path, "1", "1", trace=trace, offset="100")
    assert out == ["[103,", "106]"] * 2, f"served a value bound from the environment stale:\n{err}"
