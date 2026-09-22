"""What an upstream statement prints when the repair re-runs it is shown.

Round 29, r29s3 (CONFUSING): their new ``repair_missed_splits()`` printed what
it repaired, to stderr. Run from a cell below after an edit, the repair re-ran
it -- and the message appeared nowhere, so its false positive (a real -50%
day read as a missed split) showed only in their no-cash oracle's log. The
line stays out of the cell's own output (it belongs to another cell -- round
21), and goes in the badge, under the upstream step that printed it.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]


def test_a_message_printed_by_a_re_run_upstream_step_is_in_the_badge(nb_runner, tmp_path):
    nb_runner.create_notebook([
        "import cash\n%cash_on\n%cash_badge print\nimport sys",
        "THRESH = 0.5",
        "def repair(xs):\n    fixed = [x for x in xs if x < THRESH]\n"
        "    print('repaired splits:', fixed, file=sys.stderr)\n    return len(fixed)\n"
        "n = repair([0.1, 0.9, 0.3])",
        "print('N', n)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, "THRESH = 0.95")
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "N 3" in out, nb_runner.get_raw_output(4)
    assert "repaired splits: [0.1, 0.9, 0.3]" in out, out
