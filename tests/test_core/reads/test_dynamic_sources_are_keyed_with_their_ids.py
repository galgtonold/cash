"""A call's ``dynamic_depends_on=`` sources go into its key with their ids.

The key took the sorted tokens alone: when two sources traded versions
(x 3->5, y 5->3) the key did not move and the old result was served.
"""

from __future__ import annotations

from cash import DataSource

VERSIONS = {"x": "3", "y": "5"}


class Table(DataSource):
    def __init__(self, name):
        self.name = name

    def get_id(self) -> str:
        return "table:" + self.name

    def state_token(self):
        return VERSIONS[self.name]


def test_two_sources_that_trade_tokens_change_the_key(cash_instance):
    runs: list = []

    @cash_instance.cache(dynamic_depends_on=lambda a, b: [Table(a), Table(b)], assume_safe=True)
    def join(a, b):
        runs.append(1)  # reads the tables the way cash cannot see
        return a + b

    join("x", "y")
    join("x", "y")
    assert len(runs) == 1
    VERSIONS.update(x="5", y="3")
    try:
        join("x", "y")
    finally:
        VERSIONS.update(x="3", y="5")
    assert len(runs) == 2
