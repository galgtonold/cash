"""A test that wraps a cash function in the kernel leaves the next test a clean one.

Several tests count calls by wrapping a function of a cash module from inside
the kernel. A reused kernel keeps cash's modules from one test to the next, so
a wrapper installed only ``if not hasattr(module, 'real_...')`` was not
installed at all when an earlier test on that kernel had wrapped the same
function: the second loop test found no list of its own and its read raised
(peek returned the line of the print-mode badge that quotes the read), and a
test run twice counted on top of its first run. Which tests share a kernel
depends on how the workers' queues drain, so the failure came and went with
machine load.
"""

import pytest

from tests.test_notebook_integration.calls import test_a_call_on_a_big_argument_reads_it_once_each_side as big_arg
from tests.test_notebook_integration.calls import (
    test_a_frame_with_text_passed_to_a_call_is_read_from_its_buffers as text,
)
from tests.test_notebook_integration.loops import test_a_loop_branch_that_never_runs_reads_nothing as branch
from tests.test_notebook_integration.loops import test_a_loop_filling_lists_is_named_by_its_inputs as filling

pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]

HELPERS = "__import__('cash.notebook.control_structures.helpers').notebook.control_structures.helpers"
HASHERS = "__import__('cash.content_hashers').content_hashers"


def test_two_loop_tests_in_one_kernel_each_get_their_own_count(nb_runner):
    nb_runner.create_notebook(["x = 1"])
    nb_runner.start_kernel()
    real = nb_runner.peek(f"id({HELPERS}.update_mutated_variable_lineages)")

    # The first test's wrapper still in place, as when its test failed before undoing it.
    nb_runner.peek(f"exec({branch.COUNT!r})")
    nb_runner.peek(f"exec({filling.COUNT!r})")
    assert nb_runner.peek(f"len({filling.READ_BACK})") == "0"
    assert nb_runner.peek(f"id({HELPERS}.real_update)") == real, "a wrapper was wrapped"

    nb_runner.peek(f"exec({filling.UNCOUNT!r})")
    assert nb_runner.peek(f"id({HELPERS}.update_mutated_variable_lineages)") == real


@pytest.mark.parametrize(
    ("module", "counted"),
    [(big_arg, "reads_of_x"), (text, "pickled_items")],
    ids=["big_argument", "text_frame"],
)
def test_a_count_installed_again_starts_from_zero(nb_runner, module, counted):
    nb_runner.create_notebook(["x = 1"])
    nb_runner.start_kernel()
    nb_runner.peek(f"exec({module.COUNT!r})")
    nb_runner.peek(f"{HASHERS}.{counted}.append(1)")
    nb_runner.peek(f"exec({module.COUNT!r})")
    assert nb_runner.peek(f"len({HASHERS}.{counted})") == "0"
