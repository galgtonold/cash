"""``cash info`` shows the cap each configured tier is built with.

It used to work the caps out on its own from the top-level size alone: a
lone file backend was shown a RAM cap it does not have, and a tier's own
``max_size_bytes`` was ignored next to a "Tiers:" line naming it.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from cash import __main__ as cli
from cash.config import TierConfig, get_config


def _info(monkeypatch, capsys, **changes) -> str:
    config = dataclasses.replace(get_config(), **changes)
    monkeypatch.setattr("cash.__main__.get_config", lambda **_: config)
    cli.cmd_info(SimpleNamespace())
    out = capsys.readouterr().out
    return next(line for line in out.splitlines() if line.strip().startswith("Max size:"))


def test_a_lone_file_backend_shows_no_ram_cap(tmp_path, monkeypatch, capsys):
    line = _info(monkeypatch, capsys, backend="file", tiers=[], cache_dir=str(tmp_path))
    assert "RAM" not in line
    assert "disk" in line


def test_each_tier_shows_its_own_size(tmp_path, monkeypatch, capsys):
    tiers = [
        TierConfig(type="memory", max_size_bytes=1_000_000),
        TierConfig(type="file", cache_dir=str(tmp_path), max_size_bytes=5_000_000),
    ]
    line = _info(monkeypatch, capsys, tiers=tiers)
    assert "RAM 1 MB (1,000,000 bytes)" in line
    assert "disk 5 MB (5,000,000 bytes)" in line
    assert "auto" not in line
