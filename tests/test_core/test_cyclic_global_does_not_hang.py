"""Reading a module-level logger in a cached function hung the first call.

The key walks every argument and read global looking for a set (whose pickle
order would differ between processes). The walk had a depth limit and no
memory of where it had been, so a cyclic object graph was walked once per
PATH: ``logging.getLogger(...)`` reaches the manager, whose dict of every
logger reaches the manager again, and with the few dozen loggers of an
ordinary process the first call never returned. Every release up to 0.10.0
did this, for the most common line in a service's module header.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import textwrap

import pytest

from cash import Cash

pytestmark = [pytest.mark.core, pytest.mark.timeout(120)]


class Node:
    def __init__(self, name):
        self.name = name
        self.peers = []


def _complete_graph(n):
    nodes = [Node(i) for i in range(n)]
    for node in nodes:
        node.peers = [p for p in nodes if p is not node]
    return nodes[0]


def test_a_cyclic_argument_is_keyed_without_walking_every_path(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    calls = []

    @c.cache(assume_safe=True)
    def f(node):
        calls.append(node.name)
        return len(node.peers)

    graph = _complete_graph(12)       # 11 ** 50 paths to the depth limit
    assert f(graph) == 11
    assert f(graph) == 11
    assert calls == [0], "the second call did not hit"


def test_a_module_logger_read_in_a_cached_function_returns(tmp_path):
    """The shape that hung: a subprocess, so a regression times out one
    child process rather than a test worker."""
    script = tmp_path / "svc.py"
    script.write_text(textwrap.dedent('''
        import logging, sys
        import cash

        for i in range(40):                       # an ordinary process's loggers
            logging.getLogger(f"lib.part{i}")
        log = logging.getLogger("svc")

        @cash.cache
        def job(n):
            log.info("job %s", n)  # @cash:assume-safe
            return n * 2

        print(job(21), job(21))
    '''), encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                             timeout=60, cwd=str(tmp_path))
    except subprocess.TimeoutExpired:
        pytest.fail("the first call of a function reading a module logger did not return")
    assert out.stdout.split() == ["42", "42"], out.stderr


def test_a_logger_is_keyed_by_name(tmp_path):
    """Nothing inside a logger reaches its pickle, so none of it is walked --
    and two loggers still key apart, by name."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache(assume_safe=True)
    def name_of(logger):
        return logger.name

    assert name_of(logging.getLogger("one")) == "one"
    assert name_of(logging.getLogger("two")) == "two"
