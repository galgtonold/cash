"""Batch 276 – Feature flag and conditional logic edits.

Tests feature flags that control behavior in downstream cells.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFeatureFlagEdits:
    """Feature flag edit patterns."""

    def test_boolean_flag_edit(self, nb_runner):
        """Edit boolean feature flag, downstream behavior changes."""
        nb_runner.create_notebook([
            "USE_FANCY = True",
            "def format_name(name):\n    if USE_FANCY:\n        return f'*** {name} ***'\n    return name",
            "result = format_name('Alice')\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = *** Alice ***" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "USE_FANCY = False")
        nb_runner.run_all()
        assert "result = Alice" in nb_runner.get_output(3)

    def test_mode_string_edit(self, nb_runner):
        """Edit mode string, dispatch changes."""
        nb_runner.create_notebook([
            "MODE = 'sum'",
            "data = [10, 20, 30, 40, 50]",
            "if MODE == 'sum':\n    result = sum(data)\nelif MODE == 'count':\n    result = len(data)\nelse:\n    result = max(data)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 150" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "MODE = 'count'")
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "MODE = 'max'")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

    def test_config_object_edit(self, nb_runner):
        """Edit config object, multiple downstream cells react."""
        nb_runner.create_notebook([
            "class Config:\n    verbose = True\n    limit = 3",
            "data = list(range(10))",
            "selected = data[:Config.limit]\nif Config.verbose:\n    label = f'Selected {len(selected)} of {len(data)}'\nelse:\n    label = f'{len(selected)}'\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = Selected 3 of 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "class Config:\n    verbose = False\n    limit = 5",
        )
        nb_runner.run_all()
        assert "label = 5" in nb_runner.get_output(3)
