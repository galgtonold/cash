"""A ``CASH_*`` variable that is nearly a setting's name must not be ignored.

A wrong key in ``pyproject.toml`` and a wrong ``CASH_TIER_<N>_*`` variable are
reported; a wrong top-level variable was skipped without a word. So
``CASH_DISABLED=1`` looked like it turned cash off and did nothing, and
``CASH_MAX_CACHE_SZE=1GB`` left the cap where it was: the two settings a user
sets to protect themselves.

Unrelated ``CASH_*`` names stay ignored (other tools share the prefix), and so
do cash's own switches, which are listed in one registry
(``cash.config.env_names``). The last tests keep that registry complete.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import fields
from pathlib import Path

import pytest

import cash
from cash.config import notices
from cash.config.schema import CashConfig, check_choice
from cash.config.sources import load_env_config
from tests._scripts import run_python

SRC = Path(cash.__file__).resolve().parent


@pytest.fixture
def environment(monkeypatch):
    """Set exactly these ``CASH_*`` variables, and forget notices already said."""
    import os

    def set_only(**names):
        for name in [n for n in os.environ if n.startswith("CASH_")]:
            monkeypatch.delenv(name)
        for name, value in names.items():
            monkeypatch.setenv(name, value)

    monkeypatch.setattr(notices, "_CONFIG_NOTICES", set())
    return set_only


def _notices() -> tuple[dict, list[str]]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        loaded = load_env_config()
    return loaded, [str(w.message) for w in rec if getattr(w.message, "code", "") == "CONFIG-UNKNOWN-KEY"]


@pytest.mark.parametrize(
    "wrong, meant",
    [
        ("CASH_DISABLED", "CASH_DISABLE"),
        ("CASH_MAX_CACHE_SZE", "CASH_MAX_CACHE_SIZE"),
        ("CASH_COMPRES", "CASH_COMPRESS"),
        ("CASH_CACHEDIR", "CASH_CACHE_DIR"),
        ("CASH_PURE_PYTHN", "CASH_PURE_PYTHON"),  # a switch, not a setting: still named
    ],
)
def test_a_name_close_to_a_real_one_is_reported_with_the_name_meant(environment, wrong, meant):
    environment(**{wrong: "1"})

    loaded, said = _notices()

    assert loaded == {}, "a misspelt name must not set anything"
    assert len(said) == 1, said
    assert wrong in said[0] and f"if you meant {meant}," in said[0]


def test_it_is_said_once_however_often_the_config_is_read(environment):
    environment(CASH_DISABLED="1")
    assert len(_notices()[1]) == 1
    assert _notices()[1] == []


@pytest.mark.parametrize(
    "name",
    [
        "CASH_REGISTER",  # somebody else's
        "CASH_API_KEY",
        "CASH_HOME",
        "CASH_PURE_PYTHON",  # cash's own switches
        "CASH_TRACE",
        "CASH_VERIFY",
        "CASH_HITPATH_VERIFY",
        "CASH_TEST_REUSE_KERNEL",  # the test suite's
        "CASH_WHEEL_GATE",
    ],
)
def test_other_cash_names_stay_unreported(environment, name):
    environment(**{name: "1"})
    assert _notices() == ({}, [])


def test_real_settings_still_load_without_a_notice(environment):
    """The control: the check must not fire on what it is there to protect."""
    environment(CASH_DISABLE="1", CASH_MAX_CACHE_SIZE="1GB", CASH_CACHE_DIR="/tmp/x", CASH_TIER_0_TYPE="memory")
    loaded, said = _notices()
    assert said == []
    assert loaded["disable"] is True and loaded["max_cache_size"] == 10**9 and loaded["cache_dir"] == "/tmp/x"


def test_an_empty_misspelt_variable_is_unset_like_any_other(environment):
    environment(CASH_DISABLED="")
    assert _notices() == ({}, [])


def test_cash_info_names_the_misspelt_variables(tmp_path):
    """The whole path, as a user meets it: three settings that look set."""
    done = run_python(
        "-W",
        "always",
        "-m",
        "cash",
        "info",
        cwd=tmp_path,
        env={"CASH_DISABLED": "1", "CASH_MAX_CACHE_SZE": "1GB", "CASH_COMPRES": "1"},
    )
    out = done.stdout + done.stderr
    for wrong, meant in [
        ("CASH_DISABLED", "CASH_DISABLE"),
        ("CASH_MAX_CACHE_SZE", "CASH_MAX_CACHE_SIZE"),
        ("CASH_COMPRES", "CASH_COMPRESS"),
    ]:
        assert re.search(rf"\[CONFIG-UNKNOWN-KEY\][^\n]*{wrong},[^\n]*if you meant {meant},", out), out


# -- what "unset" means for a cap ---------------------------------------------


@pytest.mark.parametrize(
    "name, unset_means",
    [
        ("max_cache_size", "the automatic cap"),
        ("max_size_bytes", "the automatic cap"),
        ("max_memory_entries", "no limit"),
        ("max_entries", "no limit"),
    ],
)
def test_a_cap_of_zero_says_what_leaving_it_unset_does(name, unset_means):
    """Unset is not "no cap" for a size: cash then sizes the cap to the machine."""
    with pytest.raises(ValueError, match=rf"must be 1 or more \(leave it unset for {unset_means}\)"):
        check_choice(name, 0)


# -- the registry of switches --------------------------------------------------

#: A ``CASH_*`` name in a string, on a line that reads the environment or
#: names a variable for a module to read.
_ENV_READ = re.compile(r"""(?:environ|getenv|_flag\(|ENV\w*\s*=)[^\n]*?["'](CASH_[A-Z0-9_]+)["']""")


def _names_the_source_reads() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in SRC.rglob("*.py"):
        for name in _ENV_READ.findall(path.read_text(encoding="utf-8")):
            found.setdefault(name, path.relative_to(SRC).as_posix())
    return found


def test_every_variable_the_source_reads_is_a_setting_or_a_registered_switch():
    """A switch missing from ``env_names.SWITCHES`` is one typo-distance from
    a notice telling its user that it does nothing."""
    from cash.config import env_names

    read = _names_the_source_reads()
    assert {"CASH_PURE_PYTHON", "CASH_TRACE", "CASH_DISABLE", "CASH_FILE_WATCH"} <= set(read), (
        "the scan no longer finds the reads it is there to check"
    )
    settings = {f"CASH_{f.name.upper()}" for f in fields(CashConfig)}
    unknown = {
        name: where for name, where in read.items() if name not in settings and not env_names.is_known_switch(name)
    }
    assert not unknown, f"add to cash.config.env_names.SWITCHES: {unknown}"


def test_no_switch_is_mistaken_for_a_misspelt_setting(environment):
    from cash.config import env_names

    environment(**dict.fromkeys(env_names.SWITCHES, "1"))
    assert _notices()[1] == []
