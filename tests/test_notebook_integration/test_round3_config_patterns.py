"""
Batch 41: Config file patterns, environment variables, and dynamic settings
across notebook cells — common patterns in data science notebooks.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestConfigFilePatterns:
    """Test config file loading across cells."""

    def test_json_config_file(self, nb_runner, tmp_path):
        """Load JSON config file and use values across cells."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"db_host": "localhost", "db_port": 5432, "debug": true}')
        path_str = str(config_path).replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost port=5432" in nb_runner.get_output(3)

    def test_config_file_change_detected(self, nb_runner, tmp_path):
        """Change config file → re-run picks up changes."""
        config_path = tmp_path / "settings.json"
        config_path.write_text('{"mode": "dev", "batch_size": 32}')
        path_str = str(config_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import json",
            textwrap.dedent(f"""\
                with open('{path_str}') as f:
                    settings = json.load(f)
            """),
            "print(f\"mode={settings['mode']} bs={settings['batch_size']}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mode=dev bs=32" in nb_runner.get_output(3)

        # Change config
        config_path.write_text('{"mode": "prod", "batch_size": 128}')
        nb_runner.run_all()
        assert "mode=prod bs=128" in nb_runner.get_output(3)

    def test_ini_style_config(self, nb_runner, tmp_path):
        """INI-style config file using configparser."""
        ini_path = tmp_path / "app.ini"
        ini_path.write_text(
            "[database]\n"
            "host = db.example.com\n"
            "port = 3306\n"
            "\n"
            "[app]\n"
            "name = MyApp\n"
        )
        path_str = str(ini_path).replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=db.example.com name=MyApp" in nb_runner.get_output(3)


class TestEnvironmentVariables:
    """Test os.environ patterns across cells."""

    def test_env_var_access(self, nb_runner):
        """Access environment variables across cells."""
        nb_runner.create_notebook([
            "import os",
            textwrap.dedent("""\
                os.environ['MY_TEST_VAR'] = 'hello123'
                val = os.environ.get('MY_TEST_VAR', 'missing')
            """),
            "print(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=hello123" in nb_runner.get_output(3)

    def test_env_fallback(self, nb_runner):
        """Environment variable with fallback."""
        nb_runner.create_notebook([
            "import os",
            textwrap.dedent("""\
                debug_mode = os.environ.get('UNLIKELY_UNIQUE_VAR_XYZ', 'false')
                print(f"debug={debug_mode}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "debug=false" in nb_runner.get_output(2)


class TestDynamicSettings:
    """Test dynamic settings that change between runs."""

    def test_config_dict_across_cells(self, nb_runner):
        """Config dict built in one cell, used in many."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "steps=310 model=linear" in nb_runner.get_output(2)

        # Change config
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            CONFIG = {
                'learning_rate': 0.01,
                'epochs': 20,
                'batch_size': 64,
                'model': 'neural_net'
            }
        """))
        nb_runner.run_all()
        assert "steps=300 model=neural_net" in nb_runner.get_output(2)

    def test_yaml_like_nested_config(self, nb_runner):
        """YAML-like nested config pattern."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                config = {
                    'data': {
                        'train_split': 0.8,
                        'features': ['age', 'income', 'score']
                    },
                    'model': {
                        'type': 'rf',
                        'n_estimators': 100
                    }
                }
            """),
            textwrap.dedent("""\
                n_features = len(config['data']['features'])
                model_type = config['model']['type']
                print(f"features={n_features} model={model_type}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "features=3 model=rf" in nb_runner.get_output(2)


class TestMultiFileConfig:
    """Test loading from multiple config files."""

    def test_merge_two_config_files(self, nb_runner, tmp_path):
        """Load and merge two config files."""
        (tmp_path / "defaults.json").write_text('{"a": 1, "b": 2, "c": 3}')
        (tmp_path / "overrides.json").write_text('{"b": 20, "d": 40}')
        d_str = str(tmp_path / "defaults.json").replace('\\', '/')
        o_str = str(tmp_path / "overrides.json").replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "('a', 1)" in output
        assert "('b', 20)" in output  # override
        assert "('d', 40)" in output
