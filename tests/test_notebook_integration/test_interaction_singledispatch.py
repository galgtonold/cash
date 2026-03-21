"""
Interaction test: functools.singledispatch for type-based dispatch.
Tests singledispatch with multiple type registrations and
cross-cell dispatch behavior.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSingledispatch:
    """Test functools.singledispatch across cells."""

    def test_singledispatch_types(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define singledispatch function
            "from functools import singledispatch\n@singledispatch\ndef format_val(val):\n    return f'unknown:{val}'\n@format_val.register(int)\ndef _(val):\n    return f'int:{val:,}'\n@format_val.register(float)\ndef _(val):\n    return f'float:{val:.2f}'\n@format_val.register(str)\ndef _(val):\n    return f'str:\"{val}\"'\n@format_val.register(list)\ndef _(val):\n    return f'list[{len(val)}]'\nprint('format_val defined')",
            # Cell 2: dispatch various types
            "results = [\n    format_val(42000),\n    format_val(3.14159),\n    format_val('hello'),\n    format_val([1, 2, 3]),\n    format_val((1, 2)),\n]\nfor r in results:\n    print(r)",
            # Cell 3: collect
            "types_seen = len(results)\nprint(f'dispatched={types_seen}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "int:42,000" in out2
        assert "float:3.14" in out2
        assert 'str:"hello"' in out2
        assert "list[3]" in out2
        assert "unknown:(1, 2)" in out2
        out3 = nb_runner.get_output(3)
        assert "dispatched=5" in out3

    def test_singledispatch_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import singledispatch\n@singledispatch\ndef stringify(val):\n    return str(val)\n@stringify.register(int)\ndef _(val):\n    return f'N={val}'\n@stringify.register(str)\ndef _(val):\n    return f'S={val}'\nprint('stringify defined')",
            "r1 = stringify(42)\nr2 = stringify('hi')\nprint(f'r1={r1}')\nprint(f'r2={r2}')",
            "combined = r1 + '|' + r2\nprint(f'combined={combined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=N=42" in nb_runner.get_output(2)
        assert "combined=N=42|S=hi" in nb_runner.get_output(3)

        # Edit the call site to use different values
        nb_runner.set_cell_source(2, "r1 = stringify(1000)\nr2 = stringify('world')\nprint(f'r1={r1}')\nprint(f'r2={r2}')")
        nb_runner.run_cells([2, 3])
        assert "r1=N=1000" in nb_runner.get_output(2)
        assert "combined=N=1000|S=world" in nb_runner.get_output(3)

    def test_singledispatch_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import singledispatch\n@singledispatch\ndef double(val):\n    return val\n@double.register(int)\ndef _(val):\n    return val * 2\n@double.register(str)\ndef _(val):\n    return val + val\nprint('double defined')",
            "results = [double(5), double('ab'), double(3.14)]\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[10, 'abab', 3.14]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "results=[10, 'abab', 3.14]" in nb_runner.get_output(2)
