"""Which calls into user code write a file a cache hit would skip (round 22)."""

import json
import shutil

from cash.notebook.cacheability import user_callee_writing_files
from cash.notebook.purity import pure


def save_chart(fig, name):
    fig.tight_layout()
    fig.savefig(name)


def export(df, path):
    df.round(2).to_csv(path)


def report(df):
    export(df, "report.csv")  # the write is one call further down
    return len(df)


def write_config(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def log(msg):
    open("run.log", "a").write(msg + "\n")


def slow_and_logged(v):
    log("slow")
    return v * 2


def compute(v):
    return v + 1


@pure
def promised(path):
    open(path, "w").write("x")


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
