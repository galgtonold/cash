"""``create_default_config`` writes a template generated from ``CashConfig``.

It used to be hand-written: it named about half the settings, described
disk eviction as LRU, left out the ``Cash(config_path=...)`` layer, linked a
placeholder URL, and overwrote an existing file without asking.
"""

from __future__ import annotations

import os
import re
from dataclasses import fields

import pytest

tomllib = pytest.importorskip("tomllib")

from cash.config import (
    CashConfig,
    TierConfig,
    _load_toml_layer,
    _tier_key_docs,
    create_default_config,
)

_SETTING = re.compile(r"^# (\w+) = (.*)$")


def _settings(text: str) -> dict[str, str]:
    """The commented-out ``key = value`` lines of the ``[cash]`` table."""
    top = text.split("[[cash.tiers]]", 1)[0]
    return {m[1]: m[2] for line in top.splitlines() if (m := _SETTING.match(line))}


def test_the_template_names_every_setting_at_its_default(tmp_path):
    path = create_default_config(str(tmp_path / "config.toml"))
    text = open(path, encoding="utf-8").read()
    settings = _settings(text)
    public = {f.name for f in fields(CashConfig) if not f.name.startswith("_")} - {"tiers"}
    assert set(settings) == public

    defaults = CashConfig()
    toml = "\n".join(f"{k} = {v}" for k, v in settings.items() if v != "(unset)")
    parsed = tomllib.loads(toml)
    for key, value in parsed.items():
        assert value == getattr(defaults, key), key
    assert {k for k, v in settings.items() if v == "(unset)"} == {k for k in public if getattr(defaults, k) is None}


def test_the_template_changes_nothing_until_a_line_is_uncommented(tmp_path):
    path = create_default_config(str(tmp_path / "config.toml"))
    data, _outcome = _load_toml_layer(tmp_path / "config.toml")
    assert data == {}, path


def test_the_tier_example_parses_into_tiers(tmp_path):
    path = create_default_config(str(tmp_path / "config.toml"))
    text = open(path, encoding="utf-8").read()
    example = text[text.index("# [[cash.tiers]]") :]
    toml = "\n".join(line[2:] if line.startswith("# ") else "" for line in example.splitlines())
    tiers = tomllib.loads(toml)["cash"]["tiers"]
    assert [TierConfig(**t).type for t in tiers] == ["memory", "redis"]


def test_the_template_carries_the_field_docs_and_every_layer(tmp_path):
    text = open(create_default_config(str(tmp_path / "c.toml")), encoding="utf-8").read()
    assert "your-repo" not in text
    assert "config_path" in text, "the Cash(config_path=...) layer is missing"
    # The disk cap's docstring, which says how the disk tier really evicts.
    assert "(GDSF)" in text
    assert "LRU eviction" not in text


def test_every_tier_key_is_described_in_the_docstring_and_the_template(tmp_path):
    """TierConfig's Attributes section is the one description of each key:
    the API reference renders it and the template copies it."""
    docs = _tier_key_docs()
    assert list(docs) == [f.name for f in fields(TierConfig)]
    assert all(docs.values())
    text = open(create_default_config(str(tmp_path / "c.toml")), encoding="utf-8").read()
    assert "#   region: s3 tier: AWS region" in text
    assert "#   wal_mode: sqlite tier: accepted but not used" in text


def test_an_existing_file_is_kept_unless_forced(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("[cash]\ndebug = true\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_default_config(str(target))
    assert target.read_text(encoding="utf-8") == "[cash]\ndebug = true\n"

    create_default_config(str(target), force=True)
    assert "# debug = false" in target.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the user config lives under %APPDATA% on Windows")
def test_the_default_path_is_the_user_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert create_default_config() == str(tmp_path / "cash" / "config.toml")
