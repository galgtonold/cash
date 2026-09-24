"""The core-set picker's feature map must name the modules that exist now.

``select_core.py`` scores a test by the cash features it touches, found by
matching module paths against ``FEATURES``. A module with no entry counts only
as plain lines. When the ``@cash.cache`` runtime moved out of ``core.py`` into
the ``decorator/`` package, the map kept pointing at ``core`` alone, so no
feature covered the decorator's own code and the core set was picked without
it. These tests fail when a split leaves the map behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.test_selection.select_core import FEATURES, _feature_of

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
CASH = REPO_ROOT / "src" / "cash"


@pytest.mark.parametrize("prefix", sorted(FEATURES))
def test_every_feature_prefix_names_a_module_that_exists(prefix):
    assert (CASH / f"{prefix}.py").is_file() or (CASH / prefix).is_dir(), (
        f"FEATURES maps {prefix!r}, which is no longer a module or package under src/cash. "
        "Map the modules it was split into or renamed to."
    )


@pytest.mark.parametrize(
    ("module", "feature"),
    [
        ("core", "decorator"),
        ("decorator/runtime", "decorator"),
        ("decorator/cached_function", "decorator"),
        ("analysis/purity_analyzer", "decorator_purity"),
        ("analysis/purity_flow", "decorator_purity"),
        ("effect_observer", "effects"),
        ("effects", "effects"),
    ],
)
def test_split_modules_keep_their_feature(module, feature):
    assert (CASH / f"{module}.py").is_file(), module
    assert _feature_of(module) == feature
