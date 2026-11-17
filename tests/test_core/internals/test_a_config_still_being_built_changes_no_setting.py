"""A config still being built moves no generation; a change to one in use does.

The memos serve what they remembered while cash's own generation
(``state_memo.GEN``) stands. Each field set while a `CashConfig` was built
counted as a setting changed, and with no default instance, every read of
the settings outside a call builds one afresh (`active_config`): the first
class fold in a process (which asks for the analysis store's folder) moved
the generation over a hundred times and never served, as no code ran under
that config.
"""

from __future__ import annotations

from cash import Cash
from cash._active import active_config, default_cash, set_default_cash
from cash.config.resolve import get_config
from cash.config.schema import CashConfig, TierConfig
from cash.decorator import state_memo


def test_building_a_config_moves_no_generation():
    before = state_memo.GEN[0]
    CashConfig()
    TierConfig(type="memory")
    get_config()
    assert state_memo.GEN[0] == before


def test_reading_the_settings_with_no_default_instance_moves_no_generation():
    previous = default_cash()
    set_default_cash(None)
    try:
        before = state_memo.GEN[0]
        active_config()
        active_config()
        assert state_memo.GEN[0] == before
    finally:
        set_default_cash(previous)


def test_a_change_to_a_built_config_moves_the_generation():
    config, tier = CashConfig(), TierConfig(type="memory")
    before = state_memo.GEN[0]
    config.verbose = not config.verbose
    assert state_memo.GEN[0] > before
    before = state_memo.GEN[0]
    tier.max_entries = 3
    assert state_memo.GEN[0] > before


def test_a_new_default_instance_moves_the_generation(tmp_path):
    previous = default_cash()
    try:
        before = state_memo.GEN[0]
        set_default_cash(Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False))
        assert state_memo.GEN[0] > before
    finally:
        set_default_cash(previous)
