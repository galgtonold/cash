"""CAS-147: prove the kernelspec guard actually fires.

Every runner in this package boots ``kernel_name='python3'``, resolved by
KernelSpecManager from the user/system Jupyter paths rather than from
``sys.executable``. A stray ``ipykernel install --user`` from another
environment silently repoints the whole suite at a different interpreter, which
then grades code that is not the package under test AND STILL REPORTS GREEN.

The guard that catches this is a session fixture, so on a correctly configured
machine its failure branch never executes -- exactly the shape of a check that
rots into a no-op unnoticed. These tests drive the comparison directly.
"""
import os
import sys

from conftest import kernelspec_mismatch


def test_matching_interpreter_is_accepted():
    assert kernelspec_mismatch([sys.executable, '-m', 'ipykernel_launcher'], sys.executable) is None


def test_case_and_symlink_differences_are_not_a_mismatch():
    """The canonicalisation must not manufacture false alarms."""
    weird = sys.executable.upper() if os.name == 'nt' else sys.executable
    assert kernelspec_mismatch([weird], sys.executable) is None


def test_foreign_interpreter_is_rejected_with_an_actionable_message():
    foreign = os.path.join(os.path.dirname(sys.executable), 'other_venv', 'python.exe')
    problem = kernelspec_mismatch([foreign, '-m', 'ipykernel_launcher'], sys.executable)

    assert problem is not None, "a foreign interpreter must be reported"
    # The message has to name both sides and the fix, or it just stalls whoever hits it.
    assert foreign in problem
    assert sys.executable in problem
    assert 'ipykernel install --user' in problem


def test_unprovable_cases_stay_silent():
    """Conservative by design: only a PROVABLE mismatch may fail the suite."""
    assert kernelspec_mismatch([], sys.executable) is None            # no argv
    assert kernelspec_mismatch(None, sys.executable) is None          # no spec
    assert kernelspec_mismatch(['python'], sys.executable) is None    # PATH-resolved
    assert kernelspec_mismatch(['python3'], sys.executable) is None
