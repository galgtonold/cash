"""A ``dynamic_depends_on=`` resolver behind a cached caller is asked again in
the next process, and when the loader ran in a process pool.

A resolver that hands out a NEW source object pinned to the catalog's current
version (a catalog refresh that builds new handles) was asked again only by
the process that stored the caller's entry. The next process -- every
scheduled run -- checked only the source pickled with the entry, which still
answers its old version, and served the caller's stale result; so did one
process whose caller reached the loader through a ``ProcessPoolExecutor``.
The resolver call is now kept with the entry and brought back from a worker.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._files import rewrite
from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(300)]

APP = """
    import os, sys
    from concurrent.futures import ProcessPoolExecutor
    import cash

    def remote_get(key):
        # A catalog cash cannot see.
        fd = os.open(f"remote_{key}.txt", os.O_RDONLY)
        try:
            return os.read(fd, 100).decode()
        finally:
            os.close(fd)

    class Table(cash.DataSource):
        def __init__(self, name, version):
            self.name, self.version = name, version
        def get_id(self):
            return f"table:{self.name}"
        def state_token(self):
            return self.version

    def resolve(name):
        return Table(name, remote_get(f"{name}_version"))

    @cash.cache(dynamic_depends_on=resolve)
    def load(name):
        return int(remote_get(name))

    @cash.cache(dynamic_depends_on=lambda name: Table(name, remote_get(f"{name}_version")))
    def load_by_lambda(name):
        return int(remote_get(name))

    @cash.cache
    def report(name):
        return load(name) + 1

    @cash.cache
    def report_by_lambda(name):
        return load_by_lambda(name) + 1

    @cash.cache
    def report_pool(name):
        with ProcessPoolExecutor(1) as ex:
            return list(ex.map(load, [name]))[0]

    def publish(value):
        with open("remote_prices.txt", "w") as fh:
            fh.write(str(value))
        with open("remote_prices_version.txt", "w") as fh:
            fh.write(f"v{value}")

    if __name__ == "__main__":
        if sys.argv[1] == "pool":
            publish(1)
            first = report_pool("prices")
            publish(2)
            print(first, report_pool("prices"))
        else:
            print(report("prices"), report_by_lambda("prices"))
"""


def _publish(tmp_path, value):
    rewrite(tmp_path / "remote_prices.txt", str(value))
    rewrite(tmp_path / "remote_prices_version.txt", f"v{value}")


def test_the_next_process_asks_the_resolver_again(tmp_path):
    (tmp_path / "app.py").write_text(textwrap.dedent(APP), encoding="utf-8")
    _publish(tmp_path, 1)
    assert run_python("app.py", "run", cwd=tmp_path).stdout.split() == ["2", "2"]
    assert run_python("app.py", "run", cwd=tmp_path).stdout.split() == ["2", "2"]
    _publish(tmp_path, 2)
    assert run_python("app.py", "run", cwd=tmp_path).stdout.split() == ["3", "3"]


def test_a_loader_run_in_a_process_pool_is_asked_again(tmp_path):
    (tmp_path / "app.py").write_text(textwrap.dedent(APP), encoding="utf-8")
    assert run_python("app.py", "pool", cwd=tmp_path).stdout.split() == ["1", "2"]
