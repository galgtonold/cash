"""
Batch 24: Advanced cross-cell interaction torture tests.

Tests that combine multiple features simultaneously: function definitions
referencing external variables, class hierarchies with file dependencies,
decorator + module reload combos, and long multi-cell computation chains.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestCrossCellFunctionState:
    """Test functions that capture state from other cells."""

    def test_function_reads_global_config(self, nb_runner):
        """Function in cell 2 reads config dict from cell 1."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                config = {
                    'multiplier': 3,
                    'offset': 10,
                    'precision': 2
                }
            """),
            textwrap.dedent("""\
                def transform(x):
                    result = x * config['multiplier'] + config['offset']
                    return round(result, config['precision'])
            """),
            textwrap.dedent("""\
                values = [1.0, 2.5, 3.7]
                results = [transform(v) for v in values]
                print(results)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # 1*3+10=13, 2.5*3+10=17.5, 3.7*3+10=21.1
        assert "13" in output
        assert "17.5" in output
        assert "21.1" in output

        # Change config -> function results should change
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            config = {
                'multiplier': 10,
                'offset': 0,
                'precision': 1
            }
        """))
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # 1*10+0=10, 2.5*10+0=25, 3.7*10+0=37
        assert "10" in output
        assert "25" in output
        assert "37" in output

    def test_multiple_functions_sharing_state(self, nb_runner):
        """Multiple functions in different cells sharing a config."""
        nb_runner.create_notebook([
            "base_rate = 0.05",
            textwrap.dedent("""\
                def simple_interest(principal, years):
                    return principal * base_rate * years
            """),
            textwrap.dedent("""\
                def compound_interest(principal, years):
                    return principal * (1 + base_rate) ** years - principal
            """),
            textwrap.dedent("""\
                si = simple_interest(1000, 5)
                ci = compound_interest(1000, 5)
                print(f"SI={si:.2f} CI={ci:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "SI=250.00" in output
        # CI = 1000 * 1.05^5 - 1000 = 1276.28 - 1000 = 276.28
        assert "CI=276.28" in output

    def test_recursive_function_across_cells(self, nb_runner):
        """Recursive function defined in one cell, called in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                memo = {}
                def fib(n):
                    if n in memo:
                        return memo[n]
                    if n < 2:
                        return n
                    result = fib(n-1) + fib(n-2)
                    memo[n] = result
                    return result
            """),
            textwrap.dedent("""\
                result = fib(20)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6765" in nb_runner.get_output(2)


class TestClassWithFileDependency:
    """Test class definitions that interact with file operations."""

    def test_class_reads_config_file(self, nb_runner, tmp_path):
        """Class method reads from a config file."""
        config_file = tmp_path / "app_config.json"
        config_file.write_text('{"version": "1.0", "debug": false}')
        path_str = str(config_file).replace('\\', '/')

        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                import json
                class AppConfig:
                    def __init__(self):
                        with open('{path_str}') as f:
                            self._data = json.load(f)
                    @property
                    def version(self):
                        return self._data['version']
            """),
            textwrap.dedent("""\
                cfg = AppConfig()
                print(cfg.version)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "1.0" in nb_runner.get_output(2)

    def test_data_processor_with_csv(self, nb_runner, tmp_path):
        """Data processing class that reads CSV files."""
        csv_path = tmp_path / "processor_data.csv"
        csv_path.write_text("metric,value\nCPU,75\nMEM,60\nDISK,45\n")
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            textwrap.dedent("""\
                class MetricAnalyzer:
                    def __init__(self, path):
                        self.df = pd.read_csv(path)
                    def summary(self):
                        return self.df['value'].describe()
                    def max_metric(self):
                        idx = self.df['value'].idxmax()
                        return self.df.loc[idx, 'metric']
            """),
            textwrap.dedent(f"""\
                analyzer = MetricAnalyzer('{path_str}')
                print(analyzer.max_metric())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "CPU" in nb_runner.get_output(3)


class TestComplexDependencyGraphs:
    """Test complex dependency graph patterns."""

    def test_diamond_with_function_deps(self, nb_runner):
        """Diamond dependency with functions: A -> B, A -> C, B+C -> D."""
        nb_runner.create_notebook([
            "base_value = 10",
            textwrap.dedent("""\
                def path_b(x):
                    return x ** 2
                b_result = path_b(base_value)
            """),
            textwrap.dedent("""\
                def path_c(x):
                    return x * 3
                c_result = path_c(base_value)
            """),
            textwrap.dedent("""\
                final = b_result + c_result
                print(final)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "130" in nb_runner.get_output(4)  # 100 + 30

        # Change root
        nb_runner.set_cell_source(1, "base_value = 5")
        nb_runner.run_all()
        assert "40" in nb_runner.get_output(4)  # 25 + 15

    def test_wide_fan_out_fan_in(self, nb_runner):
        """Many independent computations merging into one result."""
        nb_runner.create_notebook([
            "x = 10",
            "a = x + 1",
            "b = x + 2",
            "c = x + 3",
            "d = x + 4",
            "e = x + 5",
            "total = a + b + c + d + e",
            "print(total)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 11+12+13+14+15 = 65
        assert "65" in nb_runner.get_output(8)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        # 101+102+103+104+105 = 515
        assert "515" in nb_runner.get_output(8)

    def test_chain_with_intermediate_function(self, nb_runner):
        """Chain where intermediate step uses a function."""
        nb_runner.create_notebook([
            "raw = [5, 3, 8, 1, 9, 2, 7]",
            textwrap.dedent("""\
                def clean(data):
                    return sorted([x for x in data if x > 2])
                cleaned = clean(raw)
            """),
            textwrap.dedent("""\
                def analyze(data):
                    return {'min': min(data), 'max': max(data), 'mean': sum(data)/len(data)}
                stats = analyze(cleaned)
            """),
            textwrap.dedent("""\
                print(f"min={stats['min']} max={stats['max']} mean={stats['mean']:.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        # cleaned = [3, 5, 7, 8, 9], min=3, max=9, mean=6.4
        assert "min=3" in output
        assert "max=9" in output
        assert "mean=6.4" in output


class TestMultipleRestorePhases:
    """Test complex restore-after-restart scenarios."""

    def test_function_and_data_restore(self, nb_runner):
        """Both function definitions and data restore after restart."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def process(items):
                    return [x.upper() for x in items]
            """),
            textwrap.dedent("""\
                data = ['hello', 'world', 'test']
            """),
            textwrap.dedent("""\
                result = process(data)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "HELLO" in output
        assert "WORLD" in output

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "HELLO" in output
        assert "WORLD" in output

    def test_class_instance_restore(self, nb_runner):
        """Class definition and instance restore after restart."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Counter:
                    def __init__(self, start=0):
                        self.value = start
                    def increment(self):
                        self.value += 1
                        return self.value
            """),
            textwrap.dedent("""\
                c = Counter(10)
                c.increment()
                c.increment()
                print(c.value)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "12" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "12" in nb_runner.get_output(2)
