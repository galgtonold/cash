"""Every ``CASH_*`` environment variable cash knows that is not a setting.

A setting's variable is ``CASH_<FIELD>`` for a field of `CashConfig`, or
``CASH_TIER_<N>_<FIELD>`` (`sources.load_env_config`). The names here are the
rest: switches read where they act, all over the source, for diagnosing cash,
for building it and for its test suite.

They are listed in one place so that a mistyped setting can be told from a
switch: `load_env_config` warns about an unknown ``CASH_*`` name that is one
or two letters from a name it knows (``CASH_DISABLED`` for ``CASH_DISABLE``),
and must stay silent for these. **A new switch is added to `SWITCHES`, and
nowhere else**; ``tests/test_core/config/test_a_misspelt_environment_variable_is_reported.py``
fails when the source reads a name that is neither a setting nor listed.
"""

from __future__ import annotations

__all__ = ["SWITCHES", "TOOLING_PREFIXES", "is_known_switch"]

#: The switches, by what they are for.
SWITCHES: frozenset[str] = frozenset(
    {
        # -- the C extension and the build ---------------------------------
        "CASH_PURE_PYTHON",  # leave the C extension out, at build and at run time
        "CASH_REQUIRE_EXTENSION",  # the build fails if the extension does not compile
        "CASH_CWATCH_BUDGET",
        "CASH_CWATCH_MIN_BYTES",
        # -- memos that can be turned off, to rule one out -------------------
        "CASH_ANALYSIS_MEMO",
        "CASH_ANALYSIS_STORE",
        "CASH_CLASS_MEMO",
        "CASH_DATACLASS_CODE_CACHE",
        "CASH_HITPATH",
        "CASH_STATE_MEMO",
        "CASH_FILE_WATCH",
        "CASH_NATIVE_FILE_HOOK",
        # -- verify modes: do the work both ways and compare ------------------
        "CASH_VERIFY",
        "CASH_ANALYSIS_MEMO_VERIFY",
        "CASH_CLASS_MEMO_VERIFY",
        "CASH_CWATCH_VERIFY",
        "CASH_HITPATH_VERIFY",
        "CASH_STATE_MEMO_VERIFY",
        "CASH_VERIFY_INCREMENTAL_SIM",
        "CASH_VERIFY_INCREMENTAL_SIM_LOG",
        # -- traces and logs ---------------------------------------------------
        "CASH_PHASE_TIMES",
        "CASH_PHASE_TIMES_FILE",
        "CASH_SHADOW",
        "CASH_SHADOW_LOG",
        "CASH_TRACE",
        "CASH_TRACE_FILE",
        "CASH_UNSEEN_LOG",
        # -- the docs build ----------------------------------------------------
        "CASH_CLAIMS_STRICT",
    }
)

#: Prefixes of the names tooling around cash sets for itself: the test suite
#: (``CASH_TEST_REUSE_KERNEL``, ...), the wheel gate, the kernel driver.
TOOLING_PREFIXES: tuple[str, ...] = ("CASH_TEST_", "CASH_WHEEL", "CASH_DRIVER_", "CASH_STRESS_", "CASH_REPLAY_")


def is_known_switch(name: str) -> bool:
    """Is the environment variable *name* a switch or a tooling name?"""
    return name in SWITCHES or name.startswith(TOOLING_PREFIXES)
