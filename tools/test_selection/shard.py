"""Run one shard of a test suite: ``--shard=i/n`` keeps about 1/n of the tests.

Load it by module name from the repo root::

    python -m pytest tests/test_notebook_integration -p tools.test_selection.shard --shard=2/4

A test belongs to shard ``int(sha256(node id)) % n + 1``. That depends on the
node id alone, never on collection order or on which other tests exist, so the
n shards of one collection are disjoint and together hold every test, and the
xdist workers of one shard (each of which collects on its own) agree on it.
Adding a test moves no other test to another shard.

List a shard without running it: ``--collect-only -q -n 0`` prints the node
ids it keeps.
"""

from __future__ import annotations

import hashlib

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("shard")
    group.addoption(
        "--shard",
        default=None,
        metavar="I/N",
        help="Run only shard I of N (1-based), picked by a stable hash of each node id.",
    )


def parse_shard(value: str) -> tuple[int, int]:
    """``"2/4"`` -> ``(2, 4)``. Raises ``pytest.UsageError`` on anything else."""
    try:
        index_text, count_text = value.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError:
        raise pytest.UsageError(f"--shard expects I/N, like 2/4; got {value!r}") from None
    if count < 1 or not 1 <= index <= count:
        raise pytest.UsageError(f"--shard {value}: need 1 <= I <= N")
    return index, count


def shard_of(nodeid: str, count: int) -> int:
    """The 1-based shard *nodeid* belongs to when the suite is split *count* ways."""
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count + 1


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    value = config.getoption("--shard")
    if value is None:
        return
    index, count = parse_shard(value)
    kept, dropped = [], []
    for item in items:
        (kept if shard_of(item.nodeid, count) == index else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept
