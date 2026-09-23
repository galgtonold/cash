"""Which calls into user code write a file a cache hit would skip."""

import json
import shutil

from cash.analysis import namespace_effects
from cash.analysis.namespace_effects import user_callee_writing_files
from cash.purity import pure


def save_chart(fig, name):
    fig.tight_layout()
    fig.savefig(name)


def export(df, path):
    df.round(2).to_csv(path)


def report(df):
    export(df, "report.csv")  # the write is one call further down
    return len(df)


def write_config(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def log(msg):
    open("run.log", "a", encoding="utf-8").write(msg + "\n")


def slow_and_logged(v):
    log("slow")
    return v * 2


def compute(v):
    return v + 1


@pure
def promised(path):
    open(path, "w", encoding="utf-8").write("x")


def test_a_helper_that_saves_or_exports_is_a_writer():
    assert user_callee_writing_files(save_chart) == "save_chart"
    assert user_callee_writing_files(export) == "export"
    assert user_callee_writing_files(write_config) == "write_config"


def test_the_write_is_found_through_another_user_function():
    assert user_callee_writing_files(report) == "export"


def test_an_append_or_no_write_is_not():
    assert user_callee_writing_files(log) is None
    assert user_callee_writing_files(slow_and_logged) is None
    assert user_callee_writing_files(compute) is None


def test_pure_is_taken_at_its_word():
    assert user_callee_writing_files(promised) is None


def test_installed_code_is_not_looked_into():
    assert user_callee_writing_files(json.dump) is None
    assert user_callee_writing_files(shutil.copyfile) is None
    assert user_callee_writing_files(None) is None


def deep_writer(path):
    open(path, "w", encoding="utf-8").write("x")


def chain_4(path):
    deep_writer(path)


def chain_3(path):
    chain_4(path)


def chain_2(path):
    chain_3(path)


def chain_1(path):
    chain_2(path)


def chain_top(path):
    chain_1(path)


def ring_a(path):
    ring_b(path)
    deep_writer(path)


def ring_b(path):
    ring_a(path)


def test_the_verdict_does_not_depend_on_what_was_asked_first():
    """A function reached near the depth cap cannot see a writer past it; that
    cut answer must not be what a direct question about it gets later."""
    namespace_effects._callee_write_cache.clear()
    assert user_callee_writing_files(chain_top) is None  # the writer is past the cap from here
    assert user_callee_writing_files(chain_3) == "deep_writer"
    assert user_callee_writing_files(chain_2) == "deep_writer"


def test_a_call_cycle_does_not_hide_a_writer():
    """``ring_b`` only calls ``ring_a``, which calls ``ring_b`` back and then
    the writer. Asking about ``ring_a`` first must not leave ``ring_b``
    answered from the moment ``ring_a`` was still being examined."""
    namespace_effects._callee_write_cache.clear()
    assert user_callee_writing_files(ring_a) == "deep_writer"
    assert user_callee_writing_files(ring_b) == "deep_writer"
