"""Start coverage in every Python process that has this directory on its path.

``run_baseline.py`` puts this directory on PYTHONPATH, so the pytest workers
and every Jupyter kernel they launch record coverage from interpreter start.
"""

try:
    import coverage

    coverage.process_startup()
except Exception:
    pass
