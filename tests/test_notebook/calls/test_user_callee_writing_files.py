"""Which calls into user code write a file a cache hit would skip."""

import functools
import json
import shutil

import cash
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


def counted(v):
    open("calls.log", "a", encoding="utf-8").write("X")  # @cash:assume-safe
    return v


def counted_by_its_def(v):  # @cash:assume-safe
    open("calls.log", "a", encoding="utf-8").write("X")
    return v


def counted_in_a_block(v):
    with cash.assume_safe():
        open("calls.log", "a", encoding="utf-8").write("X")
    return v


def logged_on_its_word(v):
    log("x")  # @cash:assume-safe
    return v


def counted_then_saved(v, path):
    open("calls.log", "a", encoding="utf-8").write("X")  # @cash:assume-safe
    open(path, "w", encoding="utf-8").write(str(v))


@pure
def promised(path):
    open(path, "w", encoding="utf-8").write("x")


def _logged(f):
    @functools.wraps(f)
    def wrapper(*args):
        with open("calls.log", "w", encoding="utf-8") as fh:
            fh.write(f.__name__)
        return f(*args)

    return wrapper


@_logged
def logged_compute(v):
    return v + 1


def _passes(f):
    @functools.wraps(f)
    def wrapper(*args):
        return f(*args)

    return wrapper


@_passes
def passed_export(df, path):
    df.to_csv(path)


def test_a_wrapper_that_writes_is_a_writer():
    """The decorator's own body writes: read through ``__wrapped__`` alone,
    the wrapped function's body said nothing."""
    assert user_callee_writing_files(logged_compute) == "logged_compute"


def test_a_write_under_a_wrapper_is_still_seen():
    assert user_callee_writing_files(passed_export) == "passed_export"


def test_a_helper_that_saves_or_exports_is_a_writer():
    assert user_callee_writing_files(save_chart) == "save_chart"
    assert user_callee_writing_files(export) == "export"
    assert user_callee_writing_files(write_config) == "write_config"


def test_the_write_is_found_through_another_user_function():
    assert user_callee_writing_files(report) == "export"


def test_an_append_is_a_write_a_hit_would_skip():
    """An experiment log the notebook starts afresh and reads back below
    loses the rows a cache hit skips: an append counts like a replace."""
    assert user_callee_writing_files(log) == "log"
    assert user_callee_writing_files(slow_and_logged) == "log"


def test_the_upstream_replay_does_not_count_an_append():
    """Re-fired by a replay, an append writes its rows twice; asked without
    appends, only a replacing write counts."""
    assert user_callee_writing_files(log, appends=False) is None
    assert user_callee_writing_files(slow_and_logged, appends=False) is None
    assert user_callee_writing_files(export, appends=False) == "export"


def test_no_write_is_not_a_writer():
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
    """Every function is followed however deep, so a writer five calls down
    is seen from the top, and no answer depends on the question before it."""
    namespace_effects._body_cache.clear()
    assert user_callee_writing_files(chain_top) == "deep_writer"
    assert user_callee_writing_files(chain_3) == "deep_writer"
    assert user_callee_writing_files(chain_2) == "deep_writer"


def test_a_call_cycle_does_not_hide_a_writer():
    """``ring_b`` only calls ``ring_a``, which calls ``ring_b`` back and then
    the writer. Asking about ``ring_a`` first must not leave ``ring_b``
    answered from the moment ``ring_a`` was still being examined."""
    namespace_effects._body_cache.clear()
    assert user_callee_writing_files(ring_a) == "deep_writer"
    assert user_callee_writing_files(ring_b) == "deep_writer"


def test_the_same_code_in_an_installed_package_is_not_read(tmp_path, monkeypatch):
    """Code objects compare equal across files, so the verdict for a user
    function must not answer for the same function in an installed package,
    which cash does not look into."""
    import importlib
    import os
    import sys
    import textwrap

    from cash.install_paths import normcase_path

    text = textwrap.dedent(
        """
        def dump(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("x")
        """
    )
    names = []
    for folder in ("project", "site"):
        (tmp_path / folder).mkdir()
        name = f"same_code_{folder}_{id(tmp_path)}"
        (tmp_path / folder / f"{name}.py").write_text(text, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path / folder))
        names.append(name)
    user, installed = (importlib.import_module(n) for n in names)
    for n in names:
        monkeypatch.setitem(sys.modules, n, sys.modules[n])
    assert user.dump.__code__ == installed.dump.__code__
    site = normcase_path(str(tmp_path / "site"))
    monkeypatch.setattr(
        namespace_effects, "is_user_code_file", lambda f: not normcase_path(os.path.abspath(f)).startswith(site)
    )
    namespace_effects._body_cache.clear()

    assert user_callee_writing_files(user.dump) == "dump"
    assert user_callee_writing_files(installed.dump) is None


def test_a_write_the_user_waived_in_the_helper_is_not_a_writer():
    """``# @cash:assume-safe`` on the write, on the ``def`` line, or ``with
    cash.assume_safe():`` around it: in a helper it covers every caller."""
    assert user_callee_writing_files(counted) is None
    assert user_callee_writing_files(counted_by_its_def) is None
    assert user_callee_writing_files(counted_in_a_block) is None
    assert user_callee_writing_files(logged_on_its_word) is None, "a waived call waives the helper it calls"


def test_a_waiver_covers_its_own_line_only():
    assert user_callee_writing_files(counted_then_saved) == "counted_then_saved"
