"""Config files, environment variables, feature flags and constants that drive later cells."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.stress
class TestConfigFilePatterns:
    """Test config file loading across cells."""

    def test_json_config_file(self, nb_runner, tmp_path):
        """Load JSON config file and use values across cells."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"db_host": "localhost", "db_port": 5432, "debug": true}', encoding="utf-8")
        path_str = str(config_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import json",
                textwrap.dedent(f"""\
                with open('{path_str}') as f:
                    config = json.load(f)
            """),
                textwrap.dedent("""\
                host = config['db_host']
                port = config['db_port']
                print(f"host={host} port={port}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost port=5432" in nb_runner.get_output(3)

    def test_config_file_change_detected(self, nb_runner, tmp_path):
        """Change config file → re-run picks up changes."""
        config_path = tmp_path / "settings.json"
        config_path.write_text('{"mode": "dev", "batch_size": 32}', encoding="utf-8")
        path_str = str(config_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import json",
                textwrap.dedent(f"""\
                with open('{path_str}') as f:
                    settings = json.load(f)
            """),
                "print(f\"mode={settings['mode']} bs={settings['batch_size']}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mode=dev bs=32" in nb_runner.get_output(3)

        # Change config
        config_path.write_text('{"mode": "prod", "batch_size": 128}', encoding="utf-8")
        nb_runner.run_all()
        assert "mode=prod bs=128" in nb_runner.get_output(3)

    def test_ini_style_config(self, nb_runner, tmp_path):
        """INI-style config file using configparser."""
        ini_path = tmp_path / "app.ini"
        ini_path.write_text("[database]\nhost = db.example.com\nport = 3306\n\n[app]\nname = MyApp\n", encoding="utf-8")
        path_str = str(ini_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import configparser",
                textwrap.dedent(f"""\
                config = configparser.ConfigParser()
                config.read('{path_str}')
            """),
                textwrap.dedent("""\
                host = config['database']['host']
                name = config['app']['name']
                print(f"host={host} name={name}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=db.example.com name=MyApp" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestMultiFileConfig:
    """Test loading from multiple config files."""

    def test_merge_two_config_files(self, nb_runner, tmp_path):
        """Load and merge two config files."""
        (tmp_path / "defaults.json").write_text('{"a": 1, "b": 2, "c": 3}', encoding="utf-8")
        (tmp_path / "overrides.json").write_text('{"b": 20, "d": 40}', encoding="utf-8")
        d_str = str(tmp_path / "defaults.json").replace("\\", "/")
        o_str = str(tmp_path / "overrides.json").replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import json",
                textwrap.dedent(f"""\
                with open('{d_str}') as f:
                    defaults = json.load(f)
                with open('{o_str}') as f:
                    overrides = json.load(f)
            """),
                textwrap.dedent("""\
                merged = {**defaults, **overrides}
                print(sorted(merged.items()))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "('a', 1)" in output
        assert "('b', 20)" in output  # override
        assert "('d', 40)" in output


@pytest.mark.integration
@pytest.mark.stress
class TestEnvironmentVariables:
    """Test os.environ patterns across cells."""

    def test_env_var_access(self, nb_runner):
        """Access environment variables across cells."""
        nb_runner.create_notebook(
            [
                "import os",
                textwrap.dedent("""\
                os.environ['MY_TEST_VAR'] = 'hello123'
                val = os.environ.get('MY_TEST_VAR', 'missing')
            """),
                "print(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=hello123" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestDynamicSettings:
    """Test dynamic settings that change between runs."""

    def test_config_dict_across_cells(self, nb_runner):
        """Config dict built in one cell, used in many."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                CONFIG = {
                    'learning_rate': 0.001,
                    'epochs': 10,
                    'batch_size': 32,
                    'model': 'linear'
                }
            """),
                textwrap.dedent("""\
                total_steps = CONFIG['epochs'] * (1000 // CONFIG['batch_size'])
                print(f"steps={total_steps} model={CONFIG['model']}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "steps=310 model=linear" in nb_runner.get_output(2)

        # Change config
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            CONFIG = {
                'learning_rate': 0.01,
                'epochs': 20,
                'batch_size': 64,
                'model': 'neural_net'
            }
        """),
        )
        nb_runner.run_all()
        assert "steps=300 model=neural_net" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestConfigPatternEdits:
    """Editing config-like patterns."""

    def test_edit_config_value(self, nb_runner):
        """Edit a config value and check downstream."""
        nb_runner.create_notebook(
            [
                "CONFIG = {'width': 800, 'height': 600, 'title': 'App'}",
                "area = CONFIG['width'] * CONFIG['height']\nprint(f'area = {area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 480000" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "CONFIG = {'width': 1920, 'height': 1080, 'title': 'App'}")
        nb_runner.run_all()
        assert "area = 2073600" in nb_runner.get_output(2)

    def test_edit_config_toggle(self, nb_runner):
        """Toggle a config value and check downstream."""
        nb_runner.create_notebook(
            [
                "flag_value = 1  # config toggle val",
                "result_mode = str(flag_value * 100)\nprint(f'result_mode = {result_mode}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result_mode = 100" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "flag_value = 5  # config toggle val v2")
        nb_runner.run_all()
        assert "result_mode = 500" in nb_runner.get_output(2)

    def test_multi_cell_config_cascade(self, nb_runner):
        """Config cascading through multiple cells."""
        nb_runner.create_notebook(
            [
                "base = {'rate': 0.05}  # cascade config base",
                "principal = 1000  # cascade principal",
                "interest = principal * base['rate']\ntotal = principal + interest\nprint(f'total = {total}')",
            ]
        )
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


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFeatureFlagEdits:
    """Feature flag edit patterns."""

    def test_boolean_flag_edit(self, nb_runner):
        """Edit boolean feature flag, downstream behavior changes."""
        nb_runner.create_notebook(
            [
                "USE_FANCY = True",
                "def format_name(name):\n    if USE_FANCY:\n        return f'*** {name} ***'\n    return name",
                "result = format_name('Alice')\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = *** Alice ***" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "USE_FANCY = False")
        nb_runner.run_all()
        assert "result = Alice" in nb_runner.get_output(3)

    def test_mode_string_edit(self, nb_runner):
        """Edit mode string, dispatch changes."""
        nb_runner.create_notebook(
            [
                "MODE = 'sum'",
                "data = [10, 20, 30, 40, 50]",
                "if MODE == 'sum':\n    result = sum(data)\nelif MODE == 'count':\n    result = len(data)\nelse:\n    result = max(data)\nprint(f'result = {result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "class Config:\n    verbose = True\n    limit = 3",
                "data = list(range(10))",
                "selected = data[:Config.limit]\nif Config.verbose:\n    label = f'Selected {len(selected)} of {len(data)}'\nelse:\n    label = f'{len(selected)}'\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = Selected 3 of 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "class Config:\n    verbose = False\n    limit = 5",
        )
        nb_runner.run_all()
        assert "label = 5" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestConstantEdits:
    """Editing constants used across cells."""

    def test_edit_config_constant(self, nb_runner):
        """Edit a config constant that affects computation."""
        nb_runner.create_notebook(
            [
                "MAX_RETRIES = 3  # config constant",
                "retries = list(range(MAX_RETRIES))\nprint(f'retries = {retries}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "retries = [0, 1, 2]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "MAX_RETRIES = 5  # config constant increased")
        nb_runner.run_all()
        assert "retries = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)

    def test_edit_multiple_constants(self, nb_runner):
        """Edit multiple constants at once."""
        nb_runner.create_notebook(
            [
                "WIDTH = 10\nHEIGHT = 5",
                "area = WIDTH * HEIGHT\nperim = 2 * (WIDTH + HEIGHT)\nprint(f'area={area} perim={perim}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=50 perim=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "WIDTH = 20\nHEIGHT = 10")
        nb_runner.run_all()
        assert "area=200 perim=60" in nb_runner.get_output(2)

    def test_frozen_dataclass_constant(self, nb_runner):
        """Use a frozen dataclass as a constant, edit it."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass(frozen=True)\nclass Params:\n    lr: float = 0.01\n    epochs: int = 10",
                "p = Params()\nprint(f'lr={p.lr} epochs={p.epochs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lr=0.01 epochs=10" in nb_runner.get_output(3)

        # Edit defaults
        nb_runner.set_cell_source(
            2,
            "@dataclass(frozen=True)\nclass Params:\n    lr: float = 0.001\n    epochs: int = 100",
        )
        nb_runner.set_cell_source(3, "p = Params()\nprint(f'lr={p.lr} epochs={p.epochs}')")
        nb_runner.run_all()
        assert "lr=0.001 epochs=100" in nb_runner.get_output(3)
