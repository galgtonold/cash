"""Batch 70: Complex f-strings & string formatting — cash caching with advanced formatting."""
import textwrap
import pytest


@pytest.mark.stress
class TestFStringPatterns:
    """Test complex f-string patterns across cells."""

    def test_nested_fstrings(self, nb_runner):
        """Nested f-string expressions across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = {'name': 'Alice', 'scores': [90, 85, 92]}
                avg = sum(data['scores']) / len(data['scores'])
                summary = f"Student {data['name']}: avg={avg:.1f}, {'pass' if avg >= 70 else 'fail'}"
                print(f"summary={summary}")
            """),
            textwrap.dedent("""\
                header = f"{'=' * 30}"
                report = f"{header}\\n{summary}\\n{header}"
                print(report)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Student Alice: avg=89.0, pass" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "=" * 30 in out2
        assert "Alice" in out2

    def test_format_spec_expressions(self, nb_runner):
        """Format spec with computed width and precision."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                values = [3.14159, 2.71828, 1.41421]
                width = 10
                precision = 3
                formatted = [f"{v:{width}.{precision}f}" for v in values]
                print(f"formatted={formatted}")
            """),
            textwrap.dedent("""\
                joined = ' | '.join(formatted)
                print(f"table={joined}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3.142" in nb_runner.get_output(1)
        assert " | " in nb_runner.get_output(2)

    def test_fstring_propagation(self, nb_runner):
        """F-string result propagates when upstream changes."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                name = "World"
                greeting = f"Hello, {name}!"
            """),
            textwrap.dedent("""\
                print(f"msg={greeting}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Hello, World!" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            name = "Python"
            greeting = f"Hello, {name}!"
        """))
        nb_runner.run_cells([1, 2])
        assert "msg=Hello, Python!" in nb_runner.get_output(2)


@pytest.mark.stress
class TestStringFormatPatterns:
    """Test various string formatting patterns."""

    def test_template_strings(self, nb_runner):
        """string.Template across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from string import Template

                tmpl = Template('$name scored $score out of $total')
                results = []
                for name, score in [('Alice', 92), ('Bob', 87), ('Charlie', 95)]:
                    results.append(tmpl.substitute(name=name, score=score, total=100))
                print(f"count={len(results)}")
            """),
            textwrap.dedent("""\
                for r in results:
                    print(r)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "Alice scored 92" in out2
        assert "Charlie scored 95" in out2

    def test_format_map(self, nb_runner):
        """str.format_map with dict across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                template = "{city}, {country} - Pop: {pop:,}"
                cities = [
                    {'city': 'Tokyo', 'country': 'Japan', 'pop': 13960000},
                    {'city': 'London', 'country': 'UK', 'pop': 8982000},
                    {'city': 'NYC', 'country': 'USA', 'pop': 8336000},
                ]
                formatted = [template.format_map(c) for c in cities]
            """),
            textwrap.dedent("""\
                for f in formatted:
                    print(f)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Tokyo" in out
        assert "13,960,000" in out

    def test_multiline_fstring(self, nb_runner):
        """Multi-line f-string across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                items = [('Widget', 2, 9.99), ('Gadget', 1, 24.99), ('Gizmo', 3, 4.99)]
                total = sum(qty * price for _, qty, price in items)
                receipt_lines = []
                for name, qty, price in items:
                    receipt_lines.append(f"  {name:<10} x{qty:>3} @ ${price:>6.2f} = ${qty*price:>8.2f}")
                receipt = '\\n'.join(receipt_lines)
                print(f"total=${total:.2f}")
            """),
            textwrap.dedent("""\
                print(receipt)
                print(f"items={len(receipt_lines)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=$59.94" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "Widget" in out2
        assert "items=3" in out2
