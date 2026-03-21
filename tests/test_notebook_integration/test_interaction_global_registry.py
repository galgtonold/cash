"""
Batch 309: Global registry and config patterns interaction tests.
Tests editing shared state objects that downstream cells depend on.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestGlobalRegistryInteraction:
    """Test global registry/config patterns with cache invalidation."""

    def test_global_registry_edit(self, nb_runner):
        """Editing a global list should propagate to downstream consumers."""
        nb_runner.create_notebook([
            "REGISTRY = ['alpha', 'beta']",
            "count = len(REGISTRY)\nnames = ', '.join(REGISTRY)",
            "print(f'count={count},names={names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=2" in out
        assert "names=alpha, beta" in out

        nb_runner.set_cell_source(1, "REGISTRY = ['alpha', 'beta', 'gamma', 'delta']")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=4" in out

    def test_config_class_edit(self, nb_runner):
        """Editing a config class should propagate."""
        nb_runner.create_notebook([
            (
                "class Config:\n"
                "    def __init__(self):\n"
                "        self.debug = False\n"
                "        self.level = 1\n"
                "cfg = Config()"
            ),
            "mode = 'debug' if cfg.debug else 'prod'\nlvl = cfg.level",
            "info = f'{mode}:L{lvl}'",
            "print(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=prod:L1" in out

        nb_runner.set_cell_source(1, (
            "class Config:\n"
            "    def __init__(self):\n"
            "        self.debug = True\n"
            "        self.level = 5\n"
            "cfg = Config()"
        ))
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=debug:L5" in out
