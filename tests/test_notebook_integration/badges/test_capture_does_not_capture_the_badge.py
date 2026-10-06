"""``%%capture`` captures the cell's own outputs, not cash's badge.

``%%capture`` hands its body to ``run_cell``, which is cash's, and the badge
cash drew for that body went through the capture's publisher: ``cap.outputs``
held the badges, so ``cap.outputs[0]`` -- the figure or table the user
captured -- was a badge, and ``cap.show()`` drew them again.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]


def test_the_captured_outputs_are_the_users_own(nb_runner):
    nb_runner.create_notebook(
        [
            "from IPython.display import display, HTML",
            "%%capture cap\nx = 5\ndisplay(HTML('<b>my chart</b>'))",
            "print(len(cap.outputs), 'my chart' in cap.outputs[0].data['text/html'])",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "1 True"

    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "1 True"
