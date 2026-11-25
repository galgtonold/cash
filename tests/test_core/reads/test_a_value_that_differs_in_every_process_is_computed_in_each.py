"""A value that differs in every Python process is not served to the next one.

``hash()`` of text is salted per process, and ``os.getpid()`` is the
process's own: ``hash(user_id) % n`` to pick a shard, or a worker tag,
gives another number in the next run. A cached function returned the first
process's number in every later process, with no warning, while the same
expression outside it gave this process's. Each such read is part of the
key by its value now, as an environment variable is: the next process (or
another thread, for ``threading.get_ident()``) computes its own, and a
process with the same value is served. ``hash()`` of a number is the same
everywhere and is left alone.
"""

from __future__ import annotations

import ast
import os
import threading

import pytest

from cash.effects import EffectKind, classify_call, environment_input
from tests._scripts import run_python


def _call(src: str) -> ast.Call:
    node = ast.parse(src, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


@pytest.mark.parametrize(
    ("src", "part"),
    [
        ("os.getpid()", "pid"),
        ("os.getppid()", "parent_pid"),
        ("threading.get_ident()", "thread"),
        ("threading.get_native_id()", "native_thread"),
        ("hash(user_id)", "hash_seed"),
        ("hash('alice')", "hash_seed"),
        ("hash(b'alice')", "hash_seed"),
        ("hash((region, 3))", "hash_seed"),
        ("hash(str(n))", "hash_seed"),
        ("hash(key) % 16", None),  # the call inside is the read
    ],
)
def test_a_read_of_the_process_is_an_input_of_the_key(src, part):
    node = ast.parse(src, mode="eval").body
    call = node if isinstance(node, ast.Call) else node.left
    effect = classify_call(call)
    assert effect is not None and effect.kind is EffectKind.ENVIRONMENT
    assert environment_input(call) == ((part or "hash_seed"), "")


@pytest.mark.parametrize(
    "src",
    ["hash(42)", "hash(-1)", "hash(2.5)", "hash((1, 2, 3))", "hash(2**61 - 1)", "hash(len(rows))", "hash(int(text))"],
)
def test_the_hash_of_a_number_is_the_same_everywhere(src):
    assert classify_call(_call(src)) is None


def test_a_hash_function_of_the_users_is_not_the_builtin():
    def hash(value):
        return 7

    assert classify_call(_call("hash(name)"), {"hash": hash}) is None


def test_a_name_bound_to_the_builtin_is():
    assert environment_input(_call("h(name)"), {"h": hash}) == ("hash_seed", "")
    assert environment_input(_call("pid()"), {"pid": os.getpid}) == ("pid", "")


SCRIPT = """
import os, sys, threading
import cash

def ran(name):
    fd = os.open(name + ".log", os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)

@cash.cache
def bucket(user_id):
    ran("bucket")
    return hash(user_id) % 100000

@cash.cache
def worker_tag():
    ran("worker_tag")
    return os.getpid()

@cash.cache
def shard(n):
    ran("shard")
    return hash((int(n), 3)) % 7 + hash(len("abc"))

class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y, "point"))

@cash.cache
def distinct(n):
    ran("distinct")
    return len({Point(i % 3, "a") for i in range(n)})

assert distinct(9) == 3
got = [bucket("alice"), worker_tag(), shard(5)]
got += [bucket("alice"), worker_tag(), shard(5)]
plain = [bucket.__wrapped__("alice"), worker_tag.__wrapped__(), shard.__wrapped__(5)] * 2
print("SAME" if got == plain else "DIFFERENT", got, plain)
"""


def _runs(tmp_path, name: str) -> int:
    return len((tmp_path / f"{name}.log").read_bytes())


def test_each_process_gets_its_own_hash_and_process_id(tmp_path):
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    for _ in range(3):
        out = run_python("main.py", cwd=tmp_path, env={"PYTHONHASHSEED": None}).stdout
        assert out.startswith("SAME"), out
    # Each process: one cached run, and the plain call made to compare.
    assert _runs(tmp_path, "bucket") == 3 * (1 + 1)
    assert _runs(tmp_path, "worker_tag") == 3 * (1 + 1)
    # The hash of numbers is the same in every process: computed once.
    assert _runs(tmp_path, "shard") == 1 + 3
    # A class's own ``__hash__`` says where a set puts its objects, which
    # is no result: a function building such a set is still served.
    assert _runs(tmp_path, "distinct") == 1


def test_a_fixed_hash_seed_keeps_the_result_between_processes(tmp_path):
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    for _ in range(3):
        out = run_python("main.py", cwd=tmp_path, env={"PYTHONHASHSEED": "7"}).stdout
        assert out.startswith("SAME"), out
    assert _runs(tmp_path, "bucket") == 1 + 3

    # Another seed is another input: not served the first seed's number.
    out = run_python("main.py", cwd=tmp_path, env={"PYTHONHASHSEED": "8"}).stdout
    assert out.startswith("SAME"), out
    assert _runs(tmp_path, "bucket") == 2 + 4


def test_the_result_is_served_within_the_process(disk_cash):
    calls = []

    @disk_cash.cache
    def worker_tag():
        calls.append(1)
        return os.getpid()

    @disk_cash.cache
    def bucket(user_id):
        calls.append(1)
        return hash(user_id) % 1000

    assert [worker_tag(), worker_tag()] == [os.getpid()] * 2
    assert [bucket("alice"), bucket("alice")] == [hash("alice") % 1000] * 2
    assert len(calls) == 2


def test_each_thread_gets_its_own_thread_id(disk_cash):
    @disk_cash.cache
    def whoami():
        return threading.get_ident()

    seen = {}

    def ask():
        seen[threading.get_ident()] = whoami()

    # Both alive at once: a finished thread's id may be handed out again.
    barrier = threading.Barrier(2)

    def ask_together():
        barrier.wait()
        ask()
        barrier.wait()

    workers = [threading.Thread(target=ask_together) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert whoami() == threading.get_ident()
    assert len(seen) == 2 and all(ident == answer for ident, answer in seen.items())
