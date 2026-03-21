"""Batch 197 – Global config pattern interaction tests.

Tests editing a global config dict and verifying that
downstream cells that depend on it update correctly.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestConfigPatternEdits:
    """Editing config-like patterns."""

    def test_edit_config_value(self, nb_runner):
        """Edit a config value and check downstream."""
        nb_runner.create_notebook([
            "CONFIG = {'width': 800, 'height': 600, 'title': 'App'}",
            "area = CONFIG['width'] * CONFIG['height']\nprint(f'area = {area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 480000" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1, "CONFIG = {'width': 1920, 'height': 1080, 'title': 'App'}"
        )
        nb_runner.run_all()
        assert "area = 2073600" in nb_runner.get_output(2)

    def test_edit_config_toggle(self, nb_runner):
        """Toggle a config value and check downstream."""
        nb_runner.create_notebook([
            "flag_value = 1  # config toggle val",
            "result_mode = str(flag_value * 100)\nprint(f'result_mode = {result_mode}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result_mode = 100" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "flag_value = 5  # config toggle val v2")
        nb_runner.run_all()
        assert "result_mode = 500" in nb_runner.get_output(2)

    def test_multi_cell_config_cascade(self, nb_runner):
        """Config cascading through multiple cells."""
        nb_runner.create_notebook([
            "base = {'rate': 0.05}  # cascade config base",
            "principal = 1000  # cascade principal",
            "interest = principal * base['rate']\ntotal = principal + interest\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 1050.0" in nb_runner.get_output(3)

        # Change rate
        nb_runner.set_cell_source(1, "base = {'rate': 0.10}  # cascade config base v2")
        nb_runner.run_all()
        assert "total = 1100.0" in nb_runner.get_output(3)

        # Change principal
        nb_runner.set_cell_source(2, "principal = 2000  # cascade principal v2")
        nb_runner.run_all()
        assert "total = 2200.0" in nb_runner.get_output(3)
