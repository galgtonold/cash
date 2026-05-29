"""Auto-start coverage in every Python process that imports this.

Placed on PYTHONPATH so that *both* the pytest process and every Jupyter
kernel subprocess (where cash actually runs) begin recording coverage as
soon as the interpreter boots. Relies on COVERAGE_PROCESS_START pointing at
the rcfile.
"""
try:
    import coverage
    coverage.process_startup()
except Exception:
    pass
