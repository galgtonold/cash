"""The purity report describes the USER's function, never cash's own code.

Under ``%cash_on`` the file tracker replaces ``open`` and the pandas readers
with cash shims. A cached user function that reads a file therefore resolves
its callee to ``cash.notebook.file_tracker``, and the analyzer used to walk
straight into it and report the shim's own ``_tracker._track_path(...)`` as
"likely side effects or scope mutations" -- blaming the user for cash's
instrumentation.

It only ever reproduced in an EDITABLE install. ``_is_user_code``'s fallback
is ``is_local_module``, which asks whether the module lives outside
site-packages; a released wheel does not, so the shim was rejected there and
no released user could see it. That asymmetry is exactly why it needs a test:
the dev environment is the only place it happens, so nothing about a normal
run would ever catch a regression.

This is the third time cash's own machinery has leaked inside its own
analysis boundary (the ``open`` shim once poisoned a cache key; a psutil call
once made ``/proc/meminfo`` a tracked dependency), which is why the guard is
a module check rather than a special case for this one shim.
"""
from __future__ import annotations

import warnings

from cash.notebook.file_tracker import FileDependencyRegistry
from cash.purity_analyzer import _is_user_code, get_analyzer

# A REAL cash shim, built the way the file tracker builds it. Its __module__
# is cash.notebook.file_tracker, so it is the exact callee a user's `open`
# resolves to once tracking is installed.
_SHIM = FileDependencyRegistry._create_open_handler(open, lambda *a, **k: None)


def _user_function_that_reads_a_file(path):
    """Stands in for user code whose `open` has been swapped for the shim."""
    with _SHIM(path) as fh:
        return fh.read()


def test_is_user_code_rejects_cash_modules_for_a_user_function():
    assert _is_user_code(_SHIM, "__main__") is False
    assert _is_user_code(_SHIM, "my_project.pipeline") is False


def test_is_user_code_still_recurses_when_cash_analyses_itself():
    """The exclusion must not blind cash to its own helpers internally."""
    assert _is_user_code(_SHIM, "cash.core") is True


def test_analyzer_does_not_attribute_cash_internals_to_the_user():
    report = get_analyzer().analyze(_user_function_that_reads_a_file)
    offenders = [
        i for i in report.issues
        if "cash.notebook" in str(i) or "file_tracker" in str(i) or "_track_path" in str(i)
    ]
    assert not offenders, (
        "the analyzer walked into cash's own file-tracking shim and reported it "
        f"as the user's impurity: {offenders}"
    )


def test_reading_a_file_through_the_shim_emits_no_impurity_warning():
    """End-to-end shape of the original report: a file read, no warning."""
    import cash

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @cash.cache
        def reads(path):
            with _SHIM(path) as fh:
                return len(fh.read())

        reads(__file__)

    impurity = [
        w for w in caught
        if type(w.message).__name__ == "CashImpurityWarning"
    ]
    assert not impurity, [str(w.message)[:200] for w in impurity]
