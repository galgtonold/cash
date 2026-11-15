"""A call cash writes an argument's change back for and a call it never
stores are different news, under different codes.

The write-back note (``clip_in_place changed 'a' in place; ... a hit writes
it back.``) shared its code with the warnings that say a call is never
stored. A code already shown prints only ``also on <name>`` for the next
function, so a function calling ``np.seterr`` after it showed nothing but
``also on quiet``: not that its calls run uncached, nor why.
"""

from __future__ import annotations

import sys
import textwrap
import warnings

import numpy as np
import pytest

from tests._scripts import run_python


def _shown(fn, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args)
    return [(getattr(w.message, "code", None), str(w.message)) for w in caught if hasattr(w.message, "code")]


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="what a call ran is recorded with sys.monitoring, Python 3.12+; below it the call runs every time "
    "(test_a_call_changing_its_argument_runs_every_time_below_python_3_12.py)",
)
def test_the_write_back_note_has_its_own_code(disk_cash):
    @disk_cash.cache
    def clip_in_place(a):
        np.clip(a, 0, 1, out=a)
        return float(a.sum())

    shown = _shown(clip_in_place, np.array([-1.0, 2.0, 0.5]))
    assert [code for code, _ in shown] == ["CACHE-ARGUMENT-WRITTEN-BACK"], shown
    assert "clip_in_place changed 'a' in place; cash stores the state it leaves 'a' in" in shown[0][1]
    logged = clip_in_place.cache_info()["warnings"]
    assert [entry["code"] for entry in logged] == ["CACHE-ARGUMENT-WRITTEN-BACK"], logged


def test_a_call_never_stored_after_a_write_back_note_says_why(disk_cash):
    @disk_cash.cache
    def clip_in_place(a):
        np.clip(a, 0, 1, out=a)
        return float(a.sum())

    @disk_cash.cache
    def quiet(x):
        np.seterr(all="ignore")
        return x + 1

    old = np.geterr()
    try:
        _shown(clip_in_place, np.array([-1.0, 2.0, 0.5]))
        shown = _shown(quiet, 1)
        assert quiet(1) == 2
    finally:
        np.seterr(**old)
    texts = [text for code, text in shown if code == "IMPURE-OBSERVED-EFFECTS"]
    assert len(texts) == 1, shown
    assert "also on" not in texts[0], texts[0]
    assert "np.seterr" in texts[0] and "calls run uncached" in texts[0], texts[0]
    assert quiet.cache_info()["hits"] == 0


def test_a_never_stored_call_after_an_observed_effect_elsewhere_says_why(disk_cash, tmp_path):
    """The same code under another kind of finding (a file the first call
    wrote, which a hit will not repeat) does not shorten the refusal."""
    import zipfile

    @disk_cash.cache
    def write_log(x):
        with zipfile.ZipFile(str(tmp_path / "a.zip"), "w") as z:
            names = z.namelist()
        return x + len(names)

    @disk_cash.cache
    def quiet(x):
        np.seterr(all="ignore")
        return x + 1

    old = np.geterr()
    try:
        first = _shown(write_log, 1)
        shown = _shown(quiet, 1)
    finally:
        np.seterr(**old)
    assert any(code == "IMPURE-OBSERVED-EFFECTS" for code, _ in first), first
    texts = [text for code, text in shown if code == "IMPURE-OBSERVED-EFFECTS"]
    assert len(texts) == 1 and "also on" not in texts[0] and "calls run uncached" in texts[0], shown


def test_the_same_refusal_on_a_second_function_still_shortens(disk_cash):
    @disk_cash.cache
    def quiet(x):
        np.seterr(all="ignore")
        return x + 1

    @disk_cash.cache
    def quieter(x):
        np.seterr(all="raise")
        return x + 2

    old = np.geterr()
    try:
        _shown(quiet, 1)
        shown = _shown(quieter, 1)
    finally:
        np.seterr(**old)
    assert [text for _, text in shown] == [
        "[IMPURE-OBSERVED-EFFECTS] also on quieter (quieter.cache_info()['warnings'])"
    ]


class _Model:
    def __init__(self):
        self.history = []
        self.coef_ = None


def test_a_refused_attribute_replay_names_the_line_that_blocked_it(disk_cash):
    @disk_cash.cache
    def fit(model, x):
        model.coef_ = x * 2
        model.history.append(x)
        return x

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fit(_Model(), 3)
        fit(_Model(), 3)
    why = fit.explain(_Model(), 3).details["why"]
    assert (
        "a hit cannot replay it: it also changed something else in its arguments in place: model.history.append (line"
        in why
    ), why
    logged = "\n".join(entry["message"] for entry in fit.cache_info()["warnings"])
    assert "a hit cannot replay the attributes the call bound:" in logged and "model.history.append" in logged, logged
    assert fit.cache_info()["hits"] == 0


def _record(holder, x):
    holder.append(x)


def test_a_refused_attribute_replay_names_the_held_list_a_helper_changed(disk_cash):
    """The change is made where reading the body does not see it: the list
    the attribute held, grown in place, is named instead."""
    record = _record

    @disk_cash.cache
    def fit(model, x):
        model.coef_ = x * 2
        getattr(record, "__call__")(model.history, x)
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit(_Model(), 3)
        fit(_Model(), 3)
    why = fit.explain(_Model(), 3).details["why"]
    assert "an in-place change of the list model.history held" in why, why
    assert fit.cache_info()["hits"] == 0


def test_the_cash_calls_line_gives_a_repeated_refusal_once(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import numpy as np
            import cash

            c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)

            @c.cache
            def quiet(x):
                np.seterr(all="ignore")
                return x + 1

            quiet(1)
            quiet(1)
            """
        ),
        encoding="utf-8",
    )
    out = run_python(script, tmp_path / ".cash", cwd=tmp_path, env={"CASH_VERBOSE": "1"})
    lines = [line for line in out.stderr.splitlines() if line.startswith("cash.calls:")]
    assert len(lines) == 2, out.stderr
    assert lines[1].count("which no variable holds") == 1, lines[1]
    assert "not stored last time:" in lines[1] and "not stored again, for the same reason" in lines[1], lines[1]
