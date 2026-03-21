"""Batch 153 – Global constant and config pattern interaction tests.

Tests where shared constants/config are defined in an early cell
and used by many downstream cells. Edit the config and verify
all downstream cells update correctly.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestConfigDrivenWorkflow:
    """Config dict driving multiple downstream cells."""

    def test_edit_config_value(self, nb_runner):
        """Edit config value, verify all downstream updates."""
        nb_runner.create_notebook([
            "config = {'scale': 2, 'offset': 10}",
            "a = 5 * config['scale']\nprint(f'a = {a}')",
            "b = 100 + config['offset']\nprint(f'b = {b}')",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10" in nb_runner.get_output(2)
        assert "b = 110" in nb_runner.get_output(3)
        assert "c = 120" in nb_runner.get_output(4)

        # Edit config
        nb_runner.set_cell_source(1, "config = {'scale': 10, 'offset': 0}")
        nb_runner.run_all()
        assert "a = 50" in nb_runner.get_output(2)
        assert "b = 100" in nb_runner.get_output(3)
        assert "c = 150" in nb_runner.get_output(4)

    def test_add_config_key(self, nb_runner):
        """Add a new key to config, use it downstream."""
        nb_runner.create_notebook([
            "params = {'lr': 0.01}",
            "effective_lr = params['lr'] * 10\nprint(f'lr = {effective_lr}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lr = 0.1" in nb_runner.get_output(2)

        # Add key and use it
        nb_runner.set_cell_source(1, "params = {'lr': 0.01, 'decay': 0.5}")
        nb_runner.set_cell_source(
            2,
            "effective_lr = params['lr'] * params['decay']\nprint(f'lr = {effective_lr}')",
        )
        nb_runner.run_all()
        assert "lr = 0.005" in nb_runner.get_output(2)


class TestConstantEdits:
    """Edit shared constants used by many cells."""

    def test_edit_constant_three_consumers(self, nb_runner):
        """One constant used by three cells."""
        nb_runner.create_notebook([
            "PI = 3.14",
            "circumference = 2 * PI * 5\nprint(f'circ = {circumference}')",
            "area = PI * 5 ** 2\nprint(f'area = {area}')",
            "volume = (4/3) * PI * 5 ** 3\nprint(f'vol = {volume}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "circ = " in nb_runner.get_output(2)
        assert "area = " in nb_runner.get_output(3)
        assert "vol = " in nb_runner.get_output(4)

        # Use more precise PI
        nb_runner.set_cell_source(1, "PI = 3.14159")
        nb_runner.run_all()
        # All downstream should update
        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        out4 = nb_runner.get_output(4)
        assert "3.14159" in out2 or "31.4159" in out2
        assert "area = " in out3
        assert "vol = " in out4

    def test_edit_constant_with_restart(self, nb_runner):
        """Edit constant, restart, verify restored correctly."""
        nb_runner.create_notebook([
            "MULTIPLIER = 5",
            "result = MULTIPLIER * 20\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Edit and run
        nb_runner.set_cell_source(1, "MULTIPLIER = 50")
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)

        # Restart - should restore
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)
