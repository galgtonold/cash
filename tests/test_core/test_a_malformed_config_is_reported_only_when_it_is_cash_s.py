"""A config file that is not valid TOML is reported only when it is cash's.

Nearly every project has a pyproject.toml, and most hold nothing of cash's. A
broken one without a ``[tool.cash]`` table is the project's problem, not
cash's, and a notice about it would teach people to filter cash's notices out.
A file that may hold cash settings -- in any of the shapes TOML allows -- is
reported, because every setting in it is being ignored.
"""

from __future__ import annotations

import warnings

import pytest

from cash import config as config_module

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _fresh_notices(monkeypatch):
    monkeypatch.setattr(config_module, "_CONFIG_NOTICES", set())


def _notices(path):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert config_module._load_toml_config(path) == {}
    return [str(w.message) for w in rec if "CONFIG-INVALID" in str(w.message) and "not valid TOML" in str(w.message)]


@pytest.mark.parametrize(
    "name, body",
    [
        ("pyproject.toml", '[tool.cash]\ncache_dir = "x\n'),
        ("pyproject.toml", "[tool.cash.tiers]\nfoo = \n"),
        ("pyproject.toml", "[tool]\ntool.cash.cache_dir = \n"),
        ("config.toml", "[cash]\ncache_dir = \n"),
        ("config.toml", "# a flat cash config file\ncache_dir = \n"),
    ],
    ids=["section", "subtable", "dotted-key", "standalone-section", "standalone-flat"],
)
def test_every_shape_that_holds_settings_is_reported(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    said = _notices(path)
    assert said and str(path) in said[0], said


def test_a_pyproject_without_a_cash_section_says_nothing(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[project]\nname = \n\n[tool.ruff]\nline-length = 100\n# [tool.cash] is not configured\n",
        encoding="utf-8",
    )
    assert not _notices(path)
