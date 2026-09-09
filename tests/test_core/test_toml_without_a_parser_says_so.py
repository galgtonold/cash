"""A config file cash cannot parse is announced, not ignored in silence.

`tomllib` arrived in Python 3.11. On 3.10 the job falls to `tomli`, and cash
cannot depend on it — cash has no required dependencies at all, deliberately.
So on 3.10 without `tomli`, `[tool.cash]` was found, skipped, and logged at
**debug**: every setting in the file ignored, `cache_dir` among them, with no
symptom beyond a cache in the wrong place and an escape hatch that appears to
do nothing.

Found by the CI matrix rather than by reading: the console-script test built a
throwaway venv with no `tomli` in it, and all three 3.10 jobs disagreed with
the four newer ones about where an installed tool caches.

The parser is faked out rather than the interpreter downgraded, so the branch
is reachable on every version the suite runs on — otherwise this test would
only ever run on 3.10 and only there would it catch a regression.
"""
from __future__ import annotations

import builtins
import warnings

import pytest

from cash import config as config_module

pytestmark = pytest.mark.core


@pytest.fixture
def no_toml_parser(monkeypatch):
    """Neither `tomllib` nor `tomli` importable, as on a bare 3.10."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("tomllib", "tomli"):
            raise ImportError(f"no module named {name} (faked)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    # raising=False so the fails-first control reports the missing WARNING
    # rather than a missing attribute: an ERROR proves nothing about behaviour.
    monkeypatch.setattr(config_module, "_TOML_NOTICE_GIVEN", False, raising=False)


def _a_config(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "demo"\nversion = "0"\n\n'
        '[tool.cash]\ncache_dir = "/srv/somewhere"\n',
        encoding="utf-8",
    )
    return path


def test_a_config_it_cannot_read_is_reported(no_toml_parser, tmp_path):
    """THE SILENCE: the file was found, skipped, and logged at debug."""
    path = _a_config(tmp_path)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        loaded = config_module._load_toml_config(path)

    assert loaded == {}, "the fixture is wrong: something parsed the file"
    text = "\n".join(str(w.message) for w in rec)
    assert "CONFIG-TOML-UNREADABLE" in text, f"nothing was said:\n{text}"
    assert str(path) in text, "the notice must name the file being ignored"
    assert "tomli" in text, "the notice must name the fix"


def test_it_says_so_once(no_toml_parser, tmp_path):
    """Config is re-read on every `get_config()`; the notice is not."""
    path = _a_config(tmp_path)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(4):
            config_module._load_toml_config(path)

    fired = [w for w in rec if "CONFIG-TOML-UNREADABLE" in str(w.message)]
    assert len(fired) == 1, f"fired {len(fired)} times"


def test_no_config_file_says_nothing(no_toml_parser, tmp_path):
    """The control: a missing parser only matters when there is a file.

    Most users have no config file at all, and warning them about a parser
    they do not need would be pure noise.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert config_module._load_toml_config(tmp_path / "absent.toml") == {}

    assert not [w for w in rec if "CONFIG-TOML-UNREADABLE" in str(w.message)]


def test_with_a_parser_the_file_is_read(tmp_path):
    """The other control: the ordinary path is untouched and still silent."""
    path = _a_config(tmp_path)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        loaded = config_module._load_toml_config(path)

    assert loaded == {"cache_dir": "/srv/somewhere"}
    assert not [w for w in rec if "CONFIG-TOML-UNREADABLE" in str(w.message)]
