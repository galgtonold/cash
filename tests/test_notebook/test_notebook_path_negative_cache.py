"""Negative (not-found) cache for notebook-path discovery (CAS-150).

``get_notebook_path()`` cached SUCCESS for 300s but never cached the ``None``
(not-found) result, and the upstream checker resolves the path 5-15 times per
cell.  When discovery is slow-failing (a stale/dead Jupyter runtime that blocks
on a network timeout before returning nothing — papermill / nbconvert / CI), a
single ``run_all`` re-paid that timeout dozens of times: ~78s to cache
``z = 1 + 1``.

These tests pin the fix:
  * a failed lookup is memoized with a SHORT TTL, so N cells trigger ~one probe
    per TTL window instead of 5-15 per cell (the stable, deterministic guard);
  * after the TTL a failed lookup is re-probed (a server started later is still
    picked up);
  * SUCCESS is unaffected — still resolved, still cached for the long TTL;
  * a resolvable path is NEVER shadowed by a stale negative (invalidate /
    set_notebook_path clear it immediately);
  * the "upstream tracking disabled" advisory fires once per session, not once
    per cell.
"""
from __future__ import annotations

import time
import warnings

import pytest

from cash.notebook import server_discovery as sd
from cash.notebook.upstream.checker import UpstreamChecker


@pytest.fixture(autouse=True)
def _reset_discovery_state():
    """Clear all module-level discovery caches + the once-per-session warn flag."""
    sd.invalidate_notebook_path_cache()
    sd.reset_notebook_discovery_warning()
    yield
    sd.invalidate_notebook_path_cache()
    sd.reset_notebook_discovery_warning()


@pytest.fixture
def failing_discovery(monkeypatch):
    """Model a slow-failing discovery: count probes of the underlying primitive.

    ``get_notebook_path()`` is left intact (the negative cache under test wraps
    it); only the primitives it calls are patched to fail, and the ipynbname
    probe counts invocations so a test can assert how many times discovery
    actually ran.
    """
    counter = {"n": 0}

    def _slow_fail():
        counter["n"] += 1
        return None

    monkeypatch.setattr(sd, "_try_vscode_path", lambda: None)
    monkeypatch.setattr(sd, "_try_ipynbname_path", _slow_fail)
    monkeypatch.setattr(sd, "_search_servers_for_notebook", lambda kid: None)
    return counter


# ---------------------------------------------------------------------------
# Call-count guard — the stable, deterministic proof
# ---------------------------------------------------------------------------

def test_repeated_failing_lookups_probe_once(failing_discovery):
    """Many resolves within one TTL window collapse to a single probe.

    Old behaviour: every call re-probed (grows 1:1 with call count). New: the
    negative cache dedupes them all to one.
    """
    for _ in range(15):
        assert sd.get_notebook_path() is None

    assert failing_discovery["n"] == 1, (
        f"expected 1 probe for 15 failing lookups, got {failing_discovery['n']} "
        "(negative cache not deduping)"
    )


def test_run_all_of_cells_bounded_probes(failing_discovery):
    """A simulated run_all resolving the path per cell probes ~once, not N*k.

    Exercises the REAL checker choke point (`_resolve_notebook_path`) plus two
    per-cell analysis helpers that each used to call discovery independently.
    """
    class _Shell:
        user_ns: dict = {}

    uc = UpstreamChecker(shell=_Shell())

    n_cells = 13
    for _ in range(n_cells):
        path = uc._resolve_notebook_path()
        uc._notebook_function_sources("z = 1 + 1", path)
        uc._notebook_class_sources("z = 1 + 1", path)

    # Old code: >= 3 probes/cell (2 helpers + phase-2 doubling) -> ~39-52.
    # New code: bounded to ~1 per TTL window (these run well within one window).
    assert failing_discovery["n"] <= 2, (
        f"expected <=2 probes across {n_cells} cells, got {failing_discovery['n']}"
    )


# ---------------------------------------------------------------------------
# Negative-cache TTL: not re-probed within TTL, re-probed after
# ---------------------------------------------------------------------------

def test_not_reprobed_within_ttl(failing_discovery):
    sd.get_notebook_path()
    sd.get_notebook_path()
    sd.get_notebook_path()
    assert failing_discovery["n"] == 1


def test_reprobed_after_ttl_expires(failing_discovery, monkeypatch):
    assert sd.get_notebook_path() is None
    assert failing_discovery["n"] == 1

    # Age the negative-cache timestamp past the TTL (deterministic — no sleep).
    monkeypatch.setattr(
        sd, "_negative_cache_time",
        time.monotonic() - (sd._NOTEBOOK_PATH_NEGATIVE_TTL + 0.5),
    )

    assert sd.get_notebook_path() is None
    assert failing_discovery["n"] == 2, "a later-started server must be re-probed after the TTL"


def test_negative_ttl_is_short():
    """The not-found TTL must stay far below the success TTL: long enough to
    dedupe a run_all, short enough to pick up a server started after load."""
    assert 0 < sd._NOTEBOOK_PATH_NEGATIVE_TTL <= 10
    assert sd._NOTEBOOK_PATH_NEGATIVE_TTL < sd._NOTEBOOK_PATH_CACHE_TTL


# ---------------------------------------------------------------------------
# Success still works + still cached (300s), unchanged
# ---------------------------------------------------------------------------

def test_success_resolves_and_is_cached(monkeypatch):
    probes = {"n": 0}

    def _resolve():
        probes["n"] += 1
        return "C:/nb/analysis.ipynb"

    monkeypatch.setattr(sd, "_try_vscode_path", lambda: None)
    monkeypatch.setattr(sd, "_try_ipynbname_path", _resolve)

    assert sd.get_notebook_path() == "C:/nb/analysis.ipynb"
    assert sd.get_notebook_path() == "C:/nb/analysis.ipynb"
    assert probes["n"] == 1, "success must be cached (not re-probed) — the 300s cache"


def test_success_after_failure_supersedes_negative(failing_discovery, monkeypatch):
    """A negative result must never permanently shadow a path that becomes
    resolvable — %cash_on (invalidate) gives discovery a fresh chance."""
    assert sd.get_notebook_path() is None  # negative cached

    # Server comes up; %cash_on invalidates the caches.
    sd.invalidate_notebook_path_cache()
    monkeypatch.setattr(sd, "_try_ipynbname_path", lambda: "C:/nb/late.ipynb")

    assert sd.get_notebook_path() == "C:/nb/late.ipynb", "resolvable path was shadowed by stale negative"


def test_set_notebook_path_clears_negative(failing_discovery, tmp_path):
    """An explicit set_notebook_path() (VS Code cell-ID path) overrides a prior
    not-found result immediately."""
    assert sd.get_notebook_path() is None  # negative cached

    nb = tmp_path / "explicit.ipynb"
    nb.write_text("{}", encoding="utf-8")
    sd.set_notebook_path(str(nb))

    assert sd.get_notebook_path() == str(nb)


# ---------------------------------------------------------------------------
# Warning fires once per session, not per cell
# ---------------------------------------------------------------------------

def test_warning_fires_once_per_session():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(10):
            sd.warn_notebook_not_found_once()

    discovery_warnings = [
        w for w in caught
        if issubclass(w.category, sd.CashNotebookDiscoveryWarning)
    ]
    assert len(discovery_warnings) == 1, (
        f"expected exactly one advisory for 10 failed cells, got {len(discovery_warnings)}"
    )


def test_warning_is_cashwarning_family():
    """So the documented blanket filter (filterwarnings(category=CashWarning))
    silences it."""
    from cash.exceptions import CashWarning
    assert issubclass(sd.CashNotebookDiscoveryWarning, CashWarning)


def test_checker_resolve_emits_warning_when_not_found(failing_discovery):
    class _Shell:
        user_ns: dict = {}

    uc = UpstreamChecker(shell=_Shell())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Two cells both fail discovery; only the first warns.
        uc._resolve_notebook_path()
        uc._resolve_notebook_path()

    discovery_warnings = [
        w for w in caught
        if issubclass(w.category, sd.CashNotebookDiscoveryWarning)
    ]
    assert len(discovery_warnings) == 1


def test_no_warning_when_path_resolves(monkeypatch):
    monkeypatch.setattr(sd, "_try_vscode_path", lambda: None)
    monkeypatch.setattr(sd, "_try_ipynbname_path", lambda: "C:/nb/ok.ipynb")

    class _Shell:
        user_ns: dict = {}

    uc = UpstreamChecker(shell=_Shell())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = uc._resolve_notebook_path()

    assert path == "C:/nb/ok.ipynb"
    assert not [w for w in caught if issubclass(w.category, sd.CashNotebookDiscoveryWarning)]


# ---------------------------------------------------------------------------
# The advisory is FALSE while a path-free reader is answering
# ---------------------------------------------------------------------------
#
# "upstream dependency tracking is disabled for this session" is about to be
# untrue in two topologies where get_notebook_path() legitimately returns None:
# Colab, which has never had a discoverable path, and a remote or containerised
# kernel whose JupyterLab extension is pushing while path discovery fails. The
# check runs on live frontend cells in both.


def _advisories(fn) -> int:
    """How many discovery advisories ``fn()`` emits."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return len([
        w for w in caught
        if issubclass(w.category, sd.CashNotebookDiscoveryWarning)
    ])


def test_a_pushed_snapshot_suppresses_the_disabled_advisory(monkeypatch):
    from cash.notebook import live_cells

    sd.reset_notebook_discovery_warning()
    monkeypatch.setattr(live_cells, "_store", {"seq": 3, "cells": [
        {"cell_type": "code", "id": "a", "source": "x = 1"},
    ]})

    assert _advisories(sd.warn_notebook_not_found_once) == 0, (
        "cash claimed upstream tracking was disabled while the extension was "
        "supplying the very cells the check was about to run on"
    )


def test_colab_suppresses_the_disabled_advisory(monkeypatch):
    sd.reset_notebook_discovery_warning()
    monkeypatch.setattr(sd, "_in_colab", lambda: True)

    assert _advisories(sd.warn_notebook_not_found_once) == 0, (
        "Colab has no discoverable notebook path by design, but reads its cells "
        "live from the frontend -- the advisory was never true there"
    )


def test_suppression_does_not_burn_the_once_per_session_flag(monkeypatch):
    """Suppression is about THIS MOMENT, not about the session.

    If a suppressed call set ``_warned_notebook_not_found``, a frontend that
    later stopped pushing would be silently un-warnable for the rest of the
    kernel's life -- trading a false advisory for a missing one, which is the
    worse direction.
    """
    from cash.notebook import live_cells

    sd.reset_notebook_discovery_warning()
    monkeypatch.setattr(live_cells, "_store", {"seq": 3, "cells": [
        {"cell_type": "code", "id": "a", "source": "x = 1"},
    ]})
    assert _advisories(sd.warn_notebook_not_found_once) == 0

    # The frontend goes away: the snapshot expires and nothing replaces it.
    monkeypatch.setattr(live_cells, "_store", {"seq": 3, "cells": None})
    assert _advisories(sd.warn_notebook_not_found_once) == 1, (
        "the advisory must still be available once the live reader stops "
        "answering; suppression must not consume the once-per-session flag"
    )


def test_no_live_reader_still_warns(monkeypatch):
    """The control arm: the advisory is right whenever it is true."""
    from cash.notebook import live_cells

    sd.reset_notebook_discovery_warning()
    monkeypatch.setattr(live_cells, "_store", {"seq": 0, "cells": None})
    monkeypatch.setattr(sd, "_in_colab", lambda: False)

    assert _advisories(sd.warn_notebook_not_found_once) == 1


def test_a_broken_live_reader_probe_keeps_the_advisory(monkeypatch):
    """A gate that cannot answer must not swallow the advisory, or a real
    papermill/CI session loses the one signal that its headline feature is off.
    """
    sd.reset_notebook_discovery_warning()
    monkeypatch.setattr(sd, "_in_colab", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert _advisories(sd.warn_notebook_not_found_once) == 1
