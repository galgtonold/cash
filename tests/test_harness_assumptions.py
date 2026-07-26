"""The cash internals the TEST INFRASTRUCTURE reaches into must actually exist.

Three separate harness mechanisms were found dead on 2026-07-26 (CAS-238), all
the same shape: a ``hasattr`` / ``getattr`` guard, or an env var, naming
something cash does not define. Each degraded silently instead of failing, so
the mechanism still *looked* present:

* ``cash._reset_default_for_tests`` — the docs suite's per-page reset. Never
  existed, so every doc page inherited whatever global config an earlier test
  left behind. Surfaced only when a leaked ``backend="redis"`` made a page take
  30 s instead of 0.74 s.
* ``CASH_DEFAULT_BACKEND_TYPE`` — same fixture's other half. Not a config
  field; unknown ``CASH_*`` vars are ignored by design, so it did nothing.
* ``ip._cash_magics_instance`` — the integration suite's "is cash really on?"
  probe. Never existed, so the probe silently fell back to "are the magics
  registered", a different question that is true whenever cash is merely
  imported.

A defensive guard whose subject does not exist is worse than no guard: it
reports success. These assertions are deliberately blunt — they fail the moment
a rename makes a harness mechanism inert, which is exactly when it would
otherwise start lying.

Scope: only what the harness *depends on*. Tests that assert an attribute is
ABSENT (a removal guard) are correct as written and are not covered here.
"""
from __future__ import annotations

import dataclasses

import pytest


def test_reset_session_exists():
    """``tests/docs/conftest.py`` resets the global singleton per page with it."""
    import cash

    assert callable(getattr(cash, "reset_session", None)), (
        "the docs suite resets cash state between pages with reset_session(); "
        "if it is renamed, every page silently inherits the previous test's "
        "global config (CAS-238)"
    )


def test_global_singleton_hook_exists():
    """The docs conftest and remote_source both read the live singleton."""
    import cash

    assert hasattr(cash, "_get_global_cash"), (
        "RemoteFileDataSource._effective_max_age reads the resolved config off "
        "the global singleton; without this hook the revalidation window "
        "silently stops applying"
    )


def test_auto_cache_flag_exists_on_the_magics():
    """``probe_cash_active()`` reads this to tell a real control arm apart."""
    pytest.importorskip("IPython")
    from cash.notebook.ipython.magics import CashMagics

    assert "_auto_cache_enabled" in CashMagics.__init__.__code__.co_names or hasattr(
        CashMagics, "_auto_cache_enabled"
    ), (
        "the integration suite's cash-off control arms are validated by reading "
        "_auto_cache_enabled off the registered CashMagics instance; if it is "
        "renamed the probe reports 'off' for every kernel and stops catching a "
        "control that is secretly cash-ON (CAS-238)"
    )


def test_magics_are_registered_under_their_class_name():
    """The probe looks the instance up in ``magics_manager.registry``."""
    from cash.notebook.ipython.magics import CashMagics

    assert CashMagics.__name__ == "CashMagics", (
        "probe_cash_active() finds the instance via "
        "magics_manager.registry['CashMagics'] — renaming the class breaks the "
        "lookup silently, and the probe then reports every kernel as cash-off"
    )


@pytest.mark.parametrize(
    "field",
    [
        # Set or read by fixtures and by the remote-source window.
        "backend",
        "cache_dir",
        "persist_all",
        "smart_persistence",
        "remote_revalidate_max_age_seconds",
        "min_execution_time_to_cache_seconds",
    ],
)
def test_config_fields_used_by_fixtures_are_real(field):
    """A ``CASH_*`` env var only works if it maps to a real field name.

    ``CASH_DEFAULT_BACKEND_TYPE`` looked like configuration and was inert,
    because unknown ``CASH_*`` vars are ignored by design — a deliberate
    tolerance that also swallows typos.
    """
    from cash.config import CashConfig

    names = {f.name for f in dataclasses.fields(CashConfig)}
    assert field in names, (
        f"{field!r} is not a CashConfig field, so CASH_{field.upper()} is "
        f"ignored silently and anything relying on it does nothing (CAS-238)"
    )
