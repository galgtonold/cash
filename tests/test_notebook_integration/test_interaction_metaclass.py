"""
Batch 293: Metaclass interaction tests.
Tests that editing classes using metaclasses properly invalidates
downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestMetaclassInteraction:
    """Test metaclass patterns with cache invalidation."""

    def test_registry_metaclass_edit(self, nb_runner):
        """Editing a class with a registry metaclass should propagate.
        
        Note: Registry._registry is mutated as a side-effect of class creation,
        so downstream cells must reference the classes directly (Alpha, Beta)
        to make the dependency explicit for lineage tracking.
        """
        nb_runner.create_notebook([
            (
                "class Registry(type):\n"
                "    _registry = {}\n"
                "    def __new__(mcs, name, bases, namespace):\n"
                "        cls = super().__new__(mcs, name, bases, namespace)\n"
                "        if name != 'Base':\n"
                "            mcs._registry[name] = cls\n"
                "        return cls\n"
                "\n"
                "class Base(metaclass=Registry):\n"
                "    pass"
            ),
            "class Alpha(Base):\n    value = 1\n\nclass Beta(Base):\n    value = 2",
            "total = Alpha.value + Beta.value",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=3" in out

        nb_runner.set_cell_source(2, "class Alpha(Base):\n    value = 10\n\nclass Beta(Base):\n    value = 20")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=30" in out

    def test_singleton_metaclass_edit(self, nb_runner):
        """Editing a singleton class should propagate."""
        nb_runner.create_notebook([
            (
                "class Singleton(type):\n"
                "    _instances = {}\n"
                "    def __call__(cls, *args, **kwargs):\n"
                "        if cls not in cls._instances:\n"
                "            cls._instances[cls] = super().__call__(*args, **kwargs)\n"
                "        return cls._instances[cls]"
            ),
            (
                "class AppConfig(metaclass=Singleton):\n"
                "    def __init__(self, name='default'):\n"
                "        self.name = name"
            ),
            "c1 = AppConfig('myapp')\nc2 = AppConfig('other')\nsame = c1 is c2",
            "print(f'name={c1.name},same={same}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "same=True" in out
        assert "name=myapp" in out

        # Edit to clear singleton cache
        nb_runner.set_cell_source(3, "Singleton._instances.clear()\nc1 = AppConfig('v2app')\nc2 = AppConfig('other')\nsame = c1 is c2")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "name=v2app" in out
        assert "same=True" in out
