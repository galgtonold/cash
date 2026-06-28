"""Removing/renaming an upstream variable definition should invalidate cached
downstream consumers (fresh-kernel semantics), not serve a stale value (CAS-62).

The variable stays in the live namespace and its lineage is never invalidated
because the now-empty producer cell no longer emits it, so the consumer's cached
input lineage still matches. Each xfail flips to XPASS when CAS-62 is fixed.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _edit_run(nb_runner, cells, edit_idx, new_src, run_idx):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(edit_idx, new_src)
    nb_runner.run_cell(run_idx)
    return nb_runner.get_output(run_idx)


def _assert_undefined(out):
    assert "NameError" in out or "not defined" in out, f"expected NameError, got: {out!r}"


@pytest.mark.xfail(reason="CAS-62: removed definition serves stale downstream value", strict=False)
def test_remove_definition_pass(nb_runner):
    _assert_undefined(_edit_run(nb_runner, ["y = 5", "z = y + 1", "print(z)"], 1, "pass", 3))


@pytest.mark.xfail(reason="CAS-62: commented-out definition serves stale value", strict=False)
def test_comment_out_definition(nb_runner):
    _assert_undefined(_edit_run(nb_runner, ["y = 5", "z = y + 1", "print(z)"], 1, "# y removed", 3))


@pytest.mark.xfail(reason="CAS-62: renamed definition serves stale value", strict=False)
def test_rename_definition(nb_runner):
    _assert_undefined(_edit_run(nb_runner, ["y = 5", "z = y + 1", "print(z)"], 1, "w = 5", 3))


@pytest.mark.xfail(reason="CAS-62: dropping one of two defs serves stale value", strict=False)
def test_remove_one_of_two_defs(nb_runner):
    _assert_undefined(_edit_run(nb_runner, ["a = 1\nb = 2", "s = a + b", "print(s)"], 1, "a = 1", 3))


@pytest.mark.xfail(reason="CAS-62: direct consumer also serves stale value", strict=False)
def test_remove_def_rerun_consumer_directly(nb_runner):
    _assert_undefined(_edit_run(nb_runner, ["y = 5", "z = y + 1\nprint(z)"], 1, "pass", 2))
