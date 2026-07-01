"""A user variable that shadows a builtin name (sum/list/max/id/...) must be
tracked as a real dependency, not skipped as a builtin (CAS-63).

The lineage layer skips `_BUILTIN_NAMES` so genuine builtins are never tracked
as data dependencies. That skip was unconditional, so a user variable named
`sum` was also skipped: editing its definition did not invalidate a downstream
consumer, which served a stale value. The skip is now guarded by
`not in variable_lineage` — a shadowed builtin IS tracked, a genuine builtin is
not, so genuine-builtin calls (`sum([1, 2, 3])`) are still not over-invalidated.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _edit_prop(nb_runner, name):
    nb_runner.create_notebook([f"{name} = 10", f"result = {name} + 5", "print(result)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "15" in nb_runner.get_output(3)
    nb_runner.set_cell_source(1, f"{name} = 100")
    nb_runner.run_cell(3)  # re-run consumer only, WITHOUT re-running the edited cell
    assert "105" in nb_runner.get_output(3), f"{name}: {nb_runner.get_output(3)!r}"


@pytest.mark.parametrize("name", ["sum", "list", "max", "min", "id", "type", "input",
                                   "filter", "map", "dict", "set", "str", "next", "format"])
def test_builtin_shadow_edit_propagates(nb_runner, name):
    _edit_prop(nb_runner, name)


def test_shadow_multi_hop(nb_runner):
    nb_runner.create_notebook(["id = 1", "a = id * 10", "b = a + 5\nprint(b)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "15" in nb_runner.get_output(3)
    nb_runner.set_cell_source(1, "id = 3")
    nb_runner.run_cell(3)
    assert "35" in nb_runner.get_output(3), f"got: {nb_runner.get_output(3)!r}"


def test_shadow_isolated_rerun_selfmod(nb_runner):
    # self-modifying var named after a builtin must not double on isolated re-run
    nb_runner.create_notebook(["sum = [1, 2]", "sum.append(9)\nprint(sum)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[1, 2, 9]" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    assert "[1, 2, 9]" in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


def test_genuine_builtin_not_over_invalidated(nb_runner):
    # `sum(data)` uses the real builtin — never tracked as a dependency, stays cached.
    nb_runner.create_notebook(["data = [1, 2, 3]", "result = sum(data)\nprint(result)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "6" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    assert "6" in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


def test_genuine_builtin_input_edit_propagates(nb_runner):
    # editing the real input `data` still propagates through a genuine-builtin call
    nb_runner.create_notebook(["data = [1, 2, 3]", "result = sum(data)\nprint(result)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "6" in nb_runner.get_output(2)
    nb_runner.set_cell_source(1, "data = [10, 20, 30]")
    nb_runner.run_cell(2)
    assert "60" in nb_runner.get_output(2), f"got: {nb_runner.get_output(2)!r}"
