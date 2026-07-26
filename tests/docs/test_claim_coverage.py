"""Docs that *demonstrate* caching must have that demonstration verified.

``infer_claims`` derives each cached function's expected (hits, misses) from its
call sites and checks them against ``cache_info()``. Some call sites are
invisible to it, and when *every* call site for a function is invisible the page
teaches runtime behaviour that nothing checks — it executes, reports green, and
could document a hit where cash misses. That is the mechanism behind the docs
drift this suite exists to stop: `data-engineering.md` printed a `cache_info()`
payload of `{'hits': 1, ...}` for functions that actually returned all zeros,
and shipped, because the page verified no claim at all.

**A ratchet, not a gate.** The known cases are listed below and the test fails
only when a *new* one appears, or when a listed one is fixed and the entry goes
stale. Per-page opt-out markers were the alternative and would have meant
eleven of them — friction for a weak signal, and eleven places for a real
problem to hide. One list is easier to burn down and impossible to grow
silently.

**Not in scope:** a page that *defines* a cached function without calling it.
That is a separate, already-triaged category (``unexercised_cached_functions``
plus ``_EXERCISE_REQUIRED``), where a sweep found most instances legitimate —
a signature reference, a network example that must not run. Gating on "this page
verifies no claim" conflates the two and flags thirteen pages, most of them
correctly written.
"""
from __future__ import annotations

import pytest

from tests.docs._harness import called_but_uninferable
from tests.docs.test_tutorials import ALL_DOCS, REPO_ROOT

# Known cases, by page -> {function: cause}. Two causes appear:
#
#   shadowed             the name is redefined later and every call sits above
#                        the last definition. Rebinding makes the earlier
#                        function object unreachable, so its cache_info() can
#                        never be read. A real limit of the mechanism.
#   calls-inside-body    every call is inside a function or class body that
#                        does not demonstrably run exactly once. Inference DOES
#                        now follow a body whose function is invoked exactly
#                        once at top level (the `asyncio.run(main())` shape), so
#                        what remains here genuinely cannot be counted:
#                        async-caching.md defines and runs `main` three times,
#                        so a call inside "main" cannot be attributed to one
#                        run -- the helper is shadowed exactly like the cached
#                        functions are. Calls under `if __name__ == "__main__":`
#                        are also counted as not running, because under the
#                        harness they don't.
#
# Burn this down; do not add to it without a reason in the PR.
KNOWN_UNVERIFIED: dict[str, set[str]] = {
    "docs/decorator.md": {"load_users", "slow_square"},
    "docs/migration_guide.md": {"slow_function"},
    "docs/tutorials/feature-guides/async-caching.md": {
        "f", "fetch_one", "fetch_user", "stock_quote",
    },
    "docs/tutorials/feature-guides/custom-file-sources.md": {"load_features"},
    "docs/tutorials/feature-guides/debugging-and-monitoring.md": {"expensive"},
    "docs/tutorials/feature-guides/iterator-caching.md": {"stream_records"},
    "docs/tutorials/feature-guides/production-transition.md": {
        "engineer_features", "extract", "transform",
    },
    "docs/tutorials/use-cases/data-engineering.md": {"aggregate", "extract", "normalize"},
    "docs/tutorials/use-cases/llm-api-calls.md": {"chat"},
    "docs/tutorials/use-cases/scientific-computing.md": {"simulate"},
}


def _actual() -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for path in ALL_DOCS:
        bad = called_but_uninferable(path)
        if bad:
            found[path.relative_to(REPO_ROOT).as_posix()] = bad
    return found


def test_no_new_unverified_cache_demonstrations():
    """A newly-unverifiable demonstration is a regression, not a style issue."""
    actual = _actual()
    new: list[str] = []
    for page, funcs in actual.items():
        known = KNOWN_UNVERIFIED.get(page, set())
        for name, why in sorted(funcs.items()):
            if name not in known:
                new.append(f"  {page}::{name} — {why}")
    assert not new, (
        "these cached functions are CALLED by their page but no call site is "
        "visible to claim inference, so the behaviour the page teaches is "
        "verified by nothing:\n"
        + "\n".join(new)
        + "\n\nFix by giving the examples distinct names, or by adding a call "
        "after the last definition (which also demonstrates the final "
        "configuration). Only add to KNOWN_UNVERIFIED with a reason."
    )


def test_known_list_has_no_stale_entries():
    """A fixed page must be removed from the list, or the ratchet stops biting.

    Without this, the list only ever grows and silently re-permits anything that
    was once broken.
    """
    actual = _actual()
    stale: list[str] = []
    for page, funcs in KNOWN_UNVERIFIED.items():
        found = set(actual.get(page, {}))
        for name in sorted(funcs - found):
            stale.append(f"  {page}::{name}")
    assert not stale, (
        "these entries are listed as unverifiable but are now verified — "
        "delete them from KNOWN_UNVERIFIED:\n" + "\n".join(stale)
    )


@pytest.mark.parametrize("page", sorted(KNOWN_UNVERIFIED))
def test_known_pages_still_exist(page):
    """Guard against a renamed page leaving a permanently-satisfied entry."""
    assert (REPO_ROOT / page).is_file(), (
        f"{page} is listed in KNOWN_UNVERIFIED but does not exist; the entry "
        f"can never be satisfied and hides whatever replaced the page"
    )
