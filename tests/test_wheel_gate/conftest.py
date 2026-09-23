"""Local pytest config for the wheel-gate CI shim.

Registers the ``wheel_gate`` marker here (not in pyproject) so the slow gate
lives entirely under this directory and never touches the fast-suite config.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wheel_gate: slow wheel-venv gate harness; opt-in, run explicitly",
    )
