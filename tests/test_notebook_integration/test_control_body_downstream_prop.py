"""An in-place mutation inside a control-structure body (if/with) must propagate
to a downstream cell that reads the mutated container (CAS-66)."""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def test_if_body_append_downstream(nb_runner):
    nb_runner.create_notebook([
        "items = [1, 2]",
        "if True:\n    items.append(3)",
        "print('len', len(items))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "len 3" in nb_runner.get_output(3), f"got: {nb_runner.get_output(3)!r}"


def test_with_body_append_downstream(nb_runner):
    nb_runner.create_notebook([
        "import contextlib\nitems = [1, 2]",
        "with contextlib.suppress(Exception):\n    items.append(3)",
        "print('len', len(items))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "len 3" in nb_runner.get_output(3), f"got: {nb_runner.get_output(3)!r}"


def test_if_body_append_same_cell_read(nb_runner):
    # owning cell reads the mutated value in the same cell
    nb_runner.create_notebook([
        "items = [1, 2]",
        "if True:\n    items.append(3)\nprint('owner', len(items))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "owner 3" in nb_runner.get_output(2), f"got: {nb_runner.get_output(2)!r}"


def test_branch_flip_activates_append(nb_runner):
    nb_runner.create_notebook([
        "items = [1, 2]",
        "if False:\n    items.append(3)",
        "print('len', len(items))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "len 2" in nb_runner.get_output(3)
    nb_runner.set_cell_source(2, "if True:\n    items.append(3)")
    nb_runner.run_cell(3)
    assert "len 3" in nb_runner.get_output(3), f"got: {nb_runner.get_output(3)!r}"
