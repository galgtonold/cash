"""numpy ufunc ``out=`` in-place mutation correctness (CAS-52).

``np.add(a, x, out=a)`` writes ``a`` in place via the out= kwarg. Detection was
added in two symmetric places: the mutation analyzer (all_mutated_vars -> the
no-lineage in-place reset on isolated re-run) and the standalone mutation-receiver
sets (lineage bump in runtime + simulation, so an edited out= statement
invalidates a downstream consumer)."""
import pytest
pytestmark = pytest.mark.upstream


def test_numpy_out_inplace_rerun(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np\na = np.array([1.0, 2.0, 3.0])",
        "np.add(a, 10, out=a)",
        "print(a.tolist())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[11.0, 12.0, 13.0]" in nb_runner.get_output(3)
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "[11.0, 12.0, 13.0]" in nb_runner.get_output(3), nb_runner.get_output(3)


def test_numpy_out_downstream_after_rerun(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np\na = np.array([1.0, 2.0, 3.0])",
        "np.add(a, 10, out=a)",
        "s = a.sum()\nprint(s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "36.0" in nb_runner.get_output(3)
    nb_runner.set_cell_source(2, "np.add(a, 100, out=a)")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "306.0" in nb_runner.get_output(3), nb_runner.get_output(3)


def test_numpy_out_downstream_only(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np\na = np.array([1.0, 2.0, 3.0])",
        "np.add(a, 10, out=a)",
        "s = a.sum()\nprint(s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "36.0" in nb_runner.get_output(3)
    nb_runner.set_cell_source(2, "np.add(a, 100, out=a)")
    nb_runner.run_cell(3)  # ONLY downstream
    assert "306.0" in nb_runner.get_output(3), nb_runner.get_output(3)


def test_numpy_out_pure_unchanged_no_overinvalidation(nb_runner):
    """Guard: re-running an UNCHANGED out= cell + downstream must NOT recompute
    incorrectly — value stays correct."""
    nb_runner.create_notebook([
        "import numpy as np\na = np.array([1.0, 2.0, 3.0])",
        "np.add(a, 10, out=a)",
        "s = a.sum()\nprint(s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "36.0" in nb_runner.get_output(3)
    nb_runner.run_cell(3)  # re-run downstream, nothing changed
    assert "36.0" in nb_runner.get_output(3), nb_runner.get_output(3)
