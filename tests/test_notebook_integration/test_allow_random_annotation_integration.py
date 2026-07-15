"""Real-kernel coverage for unseeded-randomness warnings / ``# @cash:allow-random``.

CAS-114. The unit twin (``tests/test_notebook/test_allow_random_annotation.py``)
asserts on the warning object via ``pytest.warns``. This file asserts the thing a
user actually experiences: that the warning **reaches the notebook's cell output**
through a real kernel, and that the directive silences it there.

That distinction is the point of this file. The warning is raised inside the
kernel process, so it can only reach the user if it survives IPython's warning
plumbing and lands in the cell's stderr stream — something no in-process test can
demonstrate. ``nb_runner`` is the only trustworthy oracle for it.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

WARNING_TEXT = "Unseeded randomness detected"


def test_unseeded_random_warning_reaches_cell_output(nb_runner):
    """The warning must actually surface to the user, not just be raised."""
    nb_runner.create_notebook([
        "import numpy as np",
        "x = np.random.rand(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(2)
    assert WARNING_TEXT in out
    assert "CashRandomnessWarning" in out
    assert "numpy.random.rand" in out
    # The escape hatch must be discoverable from the warning itself.
    assert "@cash:allow-random" in out
    # The statement still ran.
    assert "len 1000" in nb_runner.get_output(3)


def test_allow_random_suppresses_warning_in_cell_output(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:allow-random\nx = np.random.rand(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert WARNING_TEXT not in nb_runner.get_raw_output(2)
    assert "len 1000" in nb_runner.get_output(3)


def test_seeded_random_does_not_warn(nb_runner):
    """Seeding in an earlier cell must quiet a later draw — the detector's
    session state has to survive across cells to get this right."""
    nb_runner.create_notebook([
        "import numpy as np\nnp.random.seed(42)",
        "x = np.random.rand(1000)",
        "print('len', len(x))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert WARNING_TEXT not in nb_runner.get_raw_output(2)
    assert "len 1000" in nb_runner.get_output(3)


def test_loop_body_warns_once_not_per_iteration(nb_runner):
    """A random draw in a 50-iteration loop must not produce 50 warnings.

    Loop bodies carry a per-iteration context comment that differs every pass, so
    without stripping it the dedupe key would be unique per iteration and the
    notebook would be flooded.
    """
    nb_runner.create_notebook([
        "import numpy as np",
        "acc = []\nfor i in range(50):\n    acc.append(np.random.rand())",
        "print('n', len(acc))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.get_raw_output(2).count(WARNING_TEXT) == 1
    assert "n 50" in nb_runner.get_output(3)


def test_allow_random_still_caches(nb_runner):
    """The directive is advisory: the annotated statement caches exactly as it
    would without it. ``@cash:persist`` clears the cost-model floor so this
    measures the directive rather than the statement's runtime.
    """
    nb_runner.create_notebook([
        "import numpy as np\nnp.random.seed(0)",
        "# @cash:persist\n# @cash:allow-random\nx = np.random.rand(1000)\nprint('sum=', round(float(x.sum()), 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(2)

    nb_runner.run_cells([2])
    second = nb_runner.get_output(2)

    # A restored value is byte-identical; a recompute would advance the RNG and
    # print a different sum.
    assert first == second
    assert "sum=" in first
