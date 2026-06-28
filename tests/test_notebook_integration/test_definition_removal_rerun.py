"""Removing/renaming an upstream variable definition invalidates cached
downstream consumers (fresh-kernel semantics), raising NameError instead of
serving a stale value (CAS-62).

When a definition is removed, commented out, or renamed, the variable is
orphaned — no cell produces it anymore — so cash evicts it and its transitive
consumers from the namespace and tracking. A from-start re-run of a consumer
then raises ``NameError`` exactly as a fresh kernel would, instead of replaying
the stale cached value.
"""
import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _edit_expect_nameerror(nb_runner, cells, edit_idx, new_src, run_idx, match="not defined"):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(edit_idx, new_src)
    # The stale-read bug would serve the cached value with NO error; the fix
    # evicts the orphan so the consumer re-runs and raises. `match` is None when
    # the NameError surfaces only on stderr (auto-execute path) rather than in
    # the structured cell error.
    with pytest.raises(CellExecutionError, match=match):
        nb_runner.run_cell(run_idx)


def test_remove_definition_pass(nb_runner):
    _edit_expect_nameerror(nb_runner, ["y = 5", "z = y + 1", "print(z)"], 1, "pass", 3)


def test_comment_out_definition(nb_runner):
    _edit_expect_nameerror(nb_runner, ["y = 5", "z = y + 1", "print(z)"], 1, "# y removed", 3)


def test_rename_definition(nb_runner):
    _edit_expect_nameerror(nb_runner, ["y = 5", "z = y + 1", "print(z)"], 1, "w = 5", 3)


def test_remove_one_of_two_defs(nb_runner):
    _edit_expect_nameerror(nb_runner, ["a = 1\nb = 2", "s = a + b", "print(s)"], 1, "a = 1", 3)


def test_remove_def_rerun_consumer_directly(nb_runner):
    # Re-running the direct consumer surfaces the NameError on the auto-execute
    # stderr path, so assert only that it raises (not the stale value).
    _edit_expect_nameerror(nb_runner, ["y = 5", "z = y + 1\nprint(z)"], 1, "pass", 2, match=None)


def test_transitive_consumer_invalidated(nb_runner):
    # a -> b -> c ; remove a's definition, re-run only c.
    _edit_expect_nameerror(
        nb_runner, ["a = 1", "b = a * 10", "c = b + 5\nprint(c)"], 1, "pass", 3)
