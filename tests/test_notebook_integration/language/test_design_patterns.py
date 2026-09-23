"""Builders, observers, state machines, validators and fluent chains across cells."""

import textwrap

import pytest


# Config file patterns, environment variables, and dynamic settings
# across notebook cells — common patterns in data science notebooks.
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


# Method chaining interaction tests.
#
# Tests editing cells that use method chaining patterns
# (string chains, list chains, custom fluent APIs).
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMethodChainingEdits:
    """Editing method chaining patterns."""

    def test_edit_string_method_chain(self, nb_runner):
        """Edit input to a string method chain."""
        nb_runner.create_notebook(
            [
                "text = '  Hello, World!  '",
                "result = text.strip().lower().replace(',', '')\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hello world!" in nb_runner.get_output(2)

        # Change input text
        nb_runner.set_cell_source(1, "text = '  GOODBYE, WORLD!  '")
        nb_runner.run_all()
        assert "result = goodbye world!" in nb_runner.get_output(2)

    def test_edit_list_method_chain(self, nb_runner):
        """Edit list operations chain."""
        nb_runner.create_notebook(
            [
                "data = [3, 1, 4, 1, 5, 9, 2, 6]",
                "result = sorted(set(data))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5, 6, 9]" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [7, 7, 3, 3, 1, 1]")
        nb_runner.run_all()
        assert "result = [1, 3, 7]" in nb_runner.get_output(2)

    def test_edit_chain_middle_step(self, nb_runner):
        """Edit the middle step of a processing chain."""
        nb_runner.create_notebook(
            [
                "words = ['hello', 'world', 'foo', 'bar', 'baz']",
                "result = [w.upper() for w in words if len(w) > 3]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['HELLO', 'WORLD']" in nb_runner.get_output(2)

        # Change filter threshold
        nb_runner.set_cell_source(2, "result = [w.upper() for w in words if len(w) >= 3]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "FOO" in nb_runner.get_output(2)
        assert "BAR" in nb_runner.get_output(2)
        assert "BAZ" in nb_runner.get_output(2)

    def test_edit_dict_chain(self, nb_runner):
        """Edit dict used in chained transformations."""
        nb_runner.create_notebook(
            [
                "inventory = {'apple': 5, 'banana': 2, 'cherry': 8, 'date': 1}",
                "available = {k: v for k, v in inventory.items() if v > 2}\nnames = sorted(available.keys())\nprint(f'names = {names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['apple', 'cherry']" in nb_runner.get_output(2)

        # Update inventory
        nb_runner.set_cell_source(1, "inventory = {'apple': 1, 'banana': 10, 'cherry': 8, 'date': 5}")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "banana" in out
        assert "cherry" in out
        assert "date" in out


# Chained method calls and builder pattern interaction tests.
# Tests that editing builder/fluent API patterns properly invalidates
# the final built object downstream.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBuilderPatternInteraction:
    """Test builder/fluent chain patterns with cache invalidation."""

    def test_builder_pattern_edit(self, nb_runner):
        """Editing builder calls should propagate to final built object."""
        nb_runner.create_notebook(
            [
                (
                    "class QueryBuilder:\n"
                    "    def __init__(self):\n"
                    "        self._parts = []\n"
                    "    def select(self, fields):\n"
                    "        self._parts.append(f'SELECT {fields}')\n"
                    "        return self\n"
                    "    def from_table(self, table):\n"
                    "        self._parts.append(f'FROM {table}')\n"
                    "        return self\n"
                    "    def where(self, cond):\n"
                    "        self._parts.append(f'WHERE {cond}')\n"
                    "        return self\n"
                    "    def build(self):\n"
                    "        return ' '.join(self._parts)"
                ),
                "q = QueryBuilder().select('*').from_table('users').where('age > 18').build()",
                "print(f'query={q}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "query=SELECT * FROM users WHERE age > 18" in out

        nb_runner.set_cell_source(
            2, "q = QueryBuilder().select('name, email').from_table('employees').where('active = 1').build()"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "query=SELECT name, email FROM employees WHERE active = 1" in out

    def test_method_chain_with_intermediate_edit(self, nb_runner):
        """Editing intermediate chain steps should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Pipeline:\n"
                    "    def __init__(self, data):\n"
                    "        self.data = data\n"
                    "    def filter_gt(self, threshold):\n"
                    "        self.data = [x for x in self.data if x > threshold]\n"
                    "        return self\n"
                    "    def multiply(self, factor):\n"
                    "        self.data = [x * factor for x in self.data]\n"
                    "        return self\n"
                    "    def result(self):\n"
                    "        return self.data"
                ),
                "initial = [1, 5, 3, 8, 2, 7]",
                "res = Pipeline(initial).filter_gt(3).multiply(10).result()",
                "print(f'res={sorted(res)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "res=[50, 70, 80]" in out

        nb_runner.set_cell_source(2, "initial = [10, 50, 30, 80, 20, 70]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "res=[100, 200, 300, 500, 700, 800]" in out

    def test_config_builder_edit(self, nb_runner):
        """Editing config builder steps should propagate to final config."""
        nb_runner.create_notebook(
            [
                (
                    "class ConfigBuilder:\n"
                    "    def __init__(self):\n"
                    "        self._config = {}\n"
                    "    def set(self, key, val):\n"
                    "        self._config[key] = val\n"
                    "        return self\n"
                    "    def build(self):\n"
                    "        return dict(self._config)"
                ),
                "cfg = ConfigBuilder().set('host', 'localhost').set('port', 8080).build()",
                "info = f\"{cfg['host']}:{cfg['port']}\"",
                "print(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=localhost:8080" in out

        nb_runner.set_cell_source(2, "cfg = ConfigBuilder().set('host', 'prod.server.com').set('port', 443).build()")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=prod.server.com:443" in out


# Builder / fluent API pattern interaction tests.
#
# Tests editing builder-style method chains and fluent interfaces.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestBuilderEdits:
    """Editing builder pattern code."""

    def test_edit_builder_step(self, nb_runner):
        """Edit one step in a builder chain."""
        nb_runner.create_notebook(
            [
                "class QueryBuilder:\n    def __init__(self):\n        self._parts = []\n    def select(self, cols):\n        self._parts.append(f'SELECT {cols}')\n        return self\n    def where(self, cond):\n        self._parts.append(f'WHERE {cond}')\n        return self\n    def build(self):\n        return ' '.join(self._parts)",
                "q = QueryBuilder().select('*').where('id > 5').build()\nprint(f'q = {q}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "q = SELECT * WHERE id > 5" in nb_runner.get_output(2)

        # Edit the query
        nb_runner.set_cell_source(
            2,
            "q = QueryBuilder().select('name, age').where('age >= 18').build()\nprint(f'q = {q}')",
        )
        nb_runner.run_all()
        assert "q = SELECT name, age WHERE age >= 18" in nb_runner.get_output(2)

    def test_edit_builder_class(self, nb_runner):
        """Edit the builder class to add a new method."""
        nb_runner.create_notebook(
            [
                "class HtmlBuilder:\n    def __init__(self):\n        self._html = ''\n    def tag(self, name, content):\n        self._html += f'<{name}>{content}</{name}>'\n        return self\n    def build(self):\n        return self._html",
                "result = HtmlBuilder().tag('h1', 'Title').tag('p', 'Body').build()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = <h1>Title</h1><p>Body</p>" in nb_runner.get_output(2)

        # Add wrap method
        nb_runner.set_cell_source(
            1,
            "class HtmlBuilder:\n    def __init__(self):\n        self._html = ''\n    def tag(self, name, content):\n        self._html += f'<{name}>{content}</{name}>'\n        return self\n    def wrap(self, name):\n        self._html = f'<{name}>{self._html}</{name}>'\n        return self\n    def build(self):\n        return self._html",
        )
        nb_runner.set_cell_source(
            2,
            "result = HtmlBuilder().tag('h1', 'Hi').wrap('div').build()\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = <div><h1>Hi</h1></div>" in nb_runner.get_output(2)

    def test_edit_chain_length(self, nb_runner):
        """Edit a chain by adding/removing steps."""
        nb_runner.create_notebook(
            [
                "class Pipeline:\n    def __init__(self, val):\n        self.val = val\n    def add(self, n):\n        self.val += n\n        return self\n    def mul(self, n):\n        self.val *= n\n        return self\n    def get(self):\n        return self.val",
                "result = Pipeline(1).add(9).mul(2).get()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (1+9)*2 = 20
        assert "result = 20" in nb_runner.get_output(2)

        # Add more steps
        nb_runner.set_cell_source(
            2,
            "result = Pipeline(1).add(9).mul(2).add(5).mul(3).get()\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        # (1+9)*2=20, 20+5=25, 25*3=75
        assert "result = 75" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestChainedAttributeAccess:
    """chained attribute access and method chains with edits."""

    def test_chained_methods(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Builder:\n    def __init__(self):\n        self.parts = []\n    def add(self, part):\n        self.parts.append(part)\n        return self\n    def build(self):\n        return '-'.join(self.parts)",
                "result = Builder().add('a').add('b').add('c').build()\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=a-b-c" in nb_runner.get_output(2)

    def test_chained_edit_class(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Query:\n    def __init__(self):\n        self.filters = []\n    def where(self, cond):\n        self.filters.append(cond)\n        return self\n    def execute(self):\n        return f'filters={self.filters}'",
                "q = Query().where('x>5').where('y<10').execute()\nprint(f'q={q}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filters=['x>5', 'y<10']" in nb_runner.get_output(2)
        # Edit class to add limit
        nb_runner.set_cell_source(
            1,
            "class Query:\n    def __init__(self):\n        self.filters = []\n        self.limit_val = None\n    def where(self, cond):\n        self.filters.append(cond)\n        return self\n    def limit(self, n):\n        self.limit_val = n\n        return self\n    def execute(self):\n        return f'filters={self.filters} limit={self.limit_val}'",
        )
        nb_runner.set_cell_source(2, "q = Query().where('x>5').limit(100).execute()\nprint(f'q={q}')")
        nb_runner.run_all()
        assert "filters=['x>5'] limit=100" in nb_runner.get_output(2)

    def test_nested_attribute(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Inner:\n    def __init__(self, v):\n        self.value = v\nclass Outer:\n    def __init__(self, v):\n        self.inner = Inner(v)",
                "o = Outer(42)\nresult = o.inner.value\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=42" in nb_runner.get_output(2)


# Global config pattern interaction tests.
#
# Tests editing a global config dict and verifying that
# downstream cells that depend on it update correctly.
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


# Feature flag and conditional logic edits.
#
# Tests feature flags that control behavior in downstream cells.
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


# Observer/event pattern interaction tests.
#
# Tests editing cells with event handling, callback patterns,
# and verifying that edits propagate correctly.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestObserverPatternEdits:
    """Editing observer/event patterns."""

    def test_edit_event_handler(self, nb_runner):
        """Edit cell 2 to add a second handler — cash should re-execute."""
        nb_runner.create_notebook(
            [
                "class EventBus:\n    def __init__(self):\n        self.handlers = []\n    def on(self, fn):\n        self.handlers.append(fn)\n    def emit(self, data):\n        return [h(data) for h in self.handlers]",
                "bus = EventBus()\nbus.on(lambda x: x.upper())\nresults = bus.emit('hello')\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        assert "results = ['HELLO']" in nb_runner.get_output(2)

        # Edit cell 2: add a second handler
        nb_runner.set_cell_source(
            2,
            "bus = EventBus()\nbus.on(lambda x: x.upper())\nbus.on(lambda x: x[::-1])\nresults = bus.emit('hello')\nprint(f'results = {results}')",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        raw2 = nb_runner.get_raw_output(2)
        print(f"DEBUG filtered output cell 2: {repr(out2)}")
        print(f"DEBUG raw output cell 2: {repr(raw2)}")
        assert "HELLO" in out2
        assert "olleh" in out2

    def test_edit_callback_list(self, nb_runner):
        """Edit a list of callbacks."""
        nb_runner.create_notebook(
            [
                "transforms = [str.upper, str.strip]",
                "text = '  hello  '\nfor fn in transforms:\n    text = fn(text)\nprint(f'text = {text}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "text = HELLO" in nb_runner.get_output(2)

        # Change transform order
        nb_runner.set_cell_source(1, "transforms = [str.strip, str.title]")
        nb_runner.run_all()
        assert "text = Hello" in nb_runner.get_output(2)

    def test_edit_dispatch_map(self, nb_runner):
        """Edit a dispatch map (command pattern)."""
        nb_runner.create_notebook(
            [
                "dispatch = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b}",
                "result = dispatch['add'](10, 3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 13" in nb_runner.get_output(2)

        # Use sub command
        nb_runner.set_cell_source(2, "result = dispatch['sub'](10, 3)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)


# State machine interaction tests.
#
# Tests editing cells that implement state machine logic
# with transitions and verifying correct propagation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestStateMachineEdits:
    """Editing state machine patterns."""

    def test_edit_transition_table(self, nb_runner):
        """Edit state transition rules."""
        nb_runner.create_notebook(
            [
                "transitions = {'idle': 'running', 'running': 'done', 'done': 'idle'}",
                "state = 'idle'\nfor _ in range(3):\n    state = transitions[state]\nprint(f'state = {state}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "state = idle" in nb_runner.get_output(2)

        # Change transitions
        nb_runner.set_cell_source(1, "transitions = {'idle': 'running', 'running': 'paused', 'paused': 'running'}")
        nb_runner.run_all()
        assert "state = running" in nb_runner.get_output(2)

    def test_edit_initial_state(self, nb_runner):
        """Edit starting state."""
        nb_runner.create_notebook(
            [
                "rules = {'A': 'B', 'B': 'C', 'C': 'A'}",
                "current = 'A'\npath = [current]\nfor _ in range(4):\n    current = rules[current]\n    path.append(current)\nprint(f'path = {path}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "path = ['A', 'B', 'C', 'A', 'B']" in nb_runner.get_output(2)

        # Change start
        nb_runner.set_cell_source(
            2,
            "current = 'C'\npath = [current]\nfor _ in range(4):\n    current = rules[current]\n    path.append(current)\nprint(f'path = {path}')",
        )
        nb_runner.run_all()
        assert "path = ['C', 'A', 'B', 'C', 'A']" in nb_runner.get_output(2)

    def test_edit_event_sequence(self, nb_runner):
        """Edit event sequence processing."""
        nb_runner.create_notebook(
            [
                "events = ['start', 'pause', 'resume', 'stop']",
                "log = []\nfor e in events:\n    log.append(e.upper())\nprint(f'log = {log}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log = ['START', 'PAUSE', 'RESUME', 'STOP']" in nb_runner.get_output(2)

        # Change events
        nb_runner.set_cell_source(1, "events = ['init', 'process', 'complete']")
        nb_runner.run_all()
        assert "log = ['INIT', 'PROCESS', 'COMPLETE']" in nb_runner.get_output(2)


# Validator pattern interaction tests.
#
# Tests editing cells with input validation patterns
# and verifying correct error/success propagation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestValidatorPatternEdits:
    """Editing input validation patterns."""

    def test_edit_validation_rules(self, nb_runner):
        """Edit validation rules and check result."""
        nb_runner.create_notebook(
            [
                "def validate_age(age):\n    if age < 0:\n        return 'invalid: negative'\n    if age > 150:\n        return 'invalid: too large'\n    return 'valid'",
                "result = validate_age(25)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = valid" in nb_runner.get_output(2)

        # Test with invalid
        nb_runner.set_cell_source(2, "result = validate_age(-5)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = invalid: negative" in nb_runner.get_output(2)

    def test_edit_validator_function(self, nb_runner):
        """Edit the validator function itself."""
        nb_runner.create_notebook(
            [
                "def check(value):\n    return 'ok' if isinstance(value, int) else 'fail'",
                "r1 = check(42)\nprint(f'r1 = {r1}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1 = ok" in nb_runner.get_output(2)

        # Make validator stricter
        nb_runner.set_cell_source(
            1, "def check(value):\n    return 'ok' if isinstance(value, int) and value > 0 else 'fail'"
        )
        nb_runner.run_all()
        assert "r1 = ok" in nb_runner.get_output(2)

    def test_edit_multi_field_validation(self, nb_runner):
        """Edit validation for multiple fields."""
        nb_runner.create_notebook(
            [
                "def validate(name, age):\n    errors = []\n    if not name:\n        errors.append('name required')\n    if age < 0:\n        errors.append('age invalid')\n    return errors if errors else ['all good']",
                "result = validate('Alice', 30)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['all good']" in nb_runner.get_output(2)

        # Test with bad data
        nb_runner.set_cell_source(2, "result = validate('', -1)\nprint(f'result = {result}')")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name required" in out
        assert "age invalid" in out


# Chained method calls and fluent interface patterns.
#
# Tests method chaining / builder patterns with edits.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFluentInterface:
    """Fluent/builder interface edit patterns."""

    def test_builder_pattern_edit(self, nb_runner):
        """Edit builder class method, chained result changes."""
        nb_runner.create_notebook(
            [
                "class Query:\n    def __init__(self):\n        self.parts = []\n    def select(self, cols):\n        self.parts.append(f'SELECT {cols}')\n        return self\n    def where(self, cond):\n        self.parts.append(f'WHERE {cond}')\n        return self\n    def build(self):\n        return ' '.join(self.parts)",
                "q = Query().select('*').where('id > 5').build()\nprint(f'q = {q}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "q = SELECT * WHERE id > 5" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Query:\n    def __init__(self):\n        self.parts = []\n    def select(self, cols):\n        self.parts.append(f'SELECT {cols}')\n        return self\n    def where(self, cond):\n        self.parts.append(f'WHERE {cond}')\n        return self\n    def build(self):\n        return ' | '.join(self.parts)",
        )
        nb_runner.run_all()
        assert "q = SELECT * | WHERE id > 5" in nb_runner.get_output(2)

    def test_chain_edit_usage(self, nb_runner):
        """Edit the chain usage, builder class stays same."""
        nb_runner.create_notebook(
            [
                "class Pipe:\n    def __init__(self, val):\n        self.val = val\n    def add(self, n):\n        self.val += n\n        return self\n    def mul(self, n):\n        self.val *= n\n        return self\n    def result(self):\n        return self.val",
                "r = Pipe(10).add(5).mul(2).result()\nprint(f'r = {r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "r = Pipe(10).mul(3).add(5).result()\nprint(f'r = {r}')")
        nb_runner.run_all()
        assert "r = 35" in nb_runner.get_output(2)


# Assertion and validation interaction tests.
# Tests that editing validation logic or asserted conditions
# properly invalidates downstream cells.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestValidationInteraction:
    """Test validation/assertion patterns with cache invalidation."""

    def test_validation_function_edit(self, nb_runner):
        """Editing a validation function should propagate."""
        nb_runner.create_notebook(
            [
                "def validate(x):\n    return 0 <= x <= 100",
                "values = [10, 50, 150, -5, 80]",
                "valid = [v for v in values if validate(v)]",
                "count = len(valid)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=3" in out

        # Widen validation range
        nb_runner.set_cell_source(1, "def validate(x):\n    return -10 <= x <= 200")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=5" in out

    def test_schema_validation_edit(self, nb_runner):
        """Editing schema/rules should propagate to validated results."""
        nb_runner.create_notebook(
            [
                "required_keys = {'name', 'age'}",
                "records = [{'name': 'Alice', 'age': 30}, {'name': 'Bob'}, {'age': 25}]",
                "valid_records = [r for r in records if required_keys.issubset(r.keys())]",
                "count = len(valid_records)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=1" in out

        # Relax requirements
        nb_runner.set_cell_source(1, "required_keys = {'name'}")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=2" in out


# Design patterns — Observer, Strategy, Builder, State with cash caching.
@pytest.mark.stress
class TestStrategyPattern:
    """Test Strategy pattern."""

    def test_strategy_change_propagates(self, nb_runner):
        """Changing strategy function propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def formatter(x):
                    return f"${x:.2f}"
            """),
                textwrap.dedent("""\
                prices = [10, 20.5, 3.99]
                formatted = [formatter(p) for p in prices]
                print(f"formatted={formatted}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "$10.00" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def formatter(x):
                return f"EUR {x:.2f}"
        """),
        )
        nb_runner.run_all()
        assert "EUR 10.00" in nb_runner.get_output(2)


@pytest.mark.stress
class TestStateMachinePattern:
    """Test State Machine pattern."""

    def test_state_machine_change(self, nb_runner):
        """Changing state machine transitions propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class FSM:
                    def __init__(self):
                        self.state = 'A'
                        self.rules = {'A': 'B', 'B': 'C', 'C': 'A'}
                    def step(self):
                        self.state = self.rules.get(self.state, self.state)
                        return self.state

                fsm = FSM()
            """),
                textwrap.dedent("""\
                states = [fsm.step() for _ in range(6)]
                print(f"states={states}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "states=['B', 'C', 'A', 'B', 'C', 'A']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class FSM:
                def __init__(self):
                    self.state = 'X'
                    self.rules = {'X': 'Y', 'Y': 'X'}
                def step(self):
                    self.state = self.rules.get(self.state, self.state)
                    return self.state

            fsm = FSM()
        """),
        )
        nb_runner.run_all()
        assert "states=['Y', 'X', 'Y', 'X', 'Y', 'X']" in nb_runner.get_output(2)


# State machine patterns — cash caching with FSM implementations.
@pytest.mark.stress
class TestStateMachine:
    """Test state machine patterns across cells."""

    def test_fsm_propagation(self, nb_runner):
        """State machine propagates when events change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                states = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}
                current = 'green'
                steps = 3
                history = [current]
                for _ in range(steps):
                    current = states[current]
                    history.append(current)
            """),
                textwrap.dedent("""\
                print(f"history={history}")
                print(f"final={history[-1]}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final=green" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            states = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}
            current = 'green'
            steps = 6
            history = [current]
            for _ in range(steps):
                current = states[current]
                history.append(current)
        """),
        )
        nb_runner.run_cells([1, 2])
        # 6 transitions: green→yellow→red→green→yellow→red→green, final=green
        assert "final=green" in nb_runner.get_output(2)
        assert len(nb_runner.get_output(2).split("history=")[1].split("]")[0].split(",")) == 7
