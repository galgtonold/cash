"""f-strings, str.format, format_map, string.Template, alignment and pprint across cells."""

import textwrap

import pytest


@pytest.mark.stress
class TestFStringPatterns:
    """Test complex f-string patterns across cells."""

    def test_format_spec_expressions(self, nb_runner):
        """Format spec with computed width and precision."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3.142" in nb_runner.get_output(1)
        assert " | " in nb_runner.get_output(2)

    def test_fstring_propagation(self, nb_runner):
        """F-string result propagates when upstream changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                name = "World"
                greeting = f"Hello, {name}!"
            """),
                textwrap.dedent("""\
                print(f"msg={greeting}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Hello, World!" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            name = "Python"
            greeting = f"Hello, {name}!"
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "msg=Hello, Python!" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestFStringComplexEdits:
    """Editing complex f-string patterns."""

    def test_edit_fstring_expression(self, nb_runner):
        """Edit data used in f-string with embedded expression."""
        nb_runner.create_notebook(
            [
                "price = 19.99\nqty = 3",
                "total = price * qty\nprint(f'Total: ${total:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Total: $59.97" in nb_runner.get_output(2)

        # Change price
        nb_runner.set_cell_source(1, "price = 24.99\nqty = 3")
        nb_runner.run_all()
        assert "Total: $74.97" in nb_runner.get_output(2)

    def test_edit_fstring_conditional(self, nb_runner):
        """Edit data used in f-string with conditional."""
        nb_runner.create_notebook(
            [
                "score = 85",
                "grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C'\nprint(f'Score {score} => grade {grade}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Score 85 => grade B" in nb_runner.get_output(2)

        # Raise score
        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "Score 95 => grade A" in nb_runner.get_output(2)

    def test_edit_fstring_multiline(self, nb_runner):
        """Edit data used in multi-line f-string output."""
        nb_runner.create_notebook(
            [
                "name = 'Alice'\nage = 30\ncity = 'NYC'",
                "info = f'Name: {name}, Age: {age}, City: {city}'\nprint(info)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Name: Alice, Age: 30, City: NYC" in nb_runner.get_output(2)

        # Update all fields
        nb_runner.set_cell_source(1, "name = 'Bob'\nage = 25\ncity = 'LA'")
        nb_runner.run_all()
        assert "Name: Bob, Age: 25, City: LA" in nb_runner.get_output(2)

    def test_edit_fstring_nested_access(self, nb_runner):
        """Edit dict data used in f-string with nested access."""
        nb_runner.create_notebook(
            [
                "user = {'name': 'Alice', 'scores': [90, 85, 78]}",
                "avg = sum(user['scores']) / len(user['scores'])\nprint(f'{user[\"name\"]}: avg={avg:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice: avg=84.3" in nb_runner.get_output(2)

        # Change user data
        nb_runner.set_cell_source(1, "user = {'name': 'Bob', 'scores': [100, 95, 90]}")
        nb_runner.run_all()
        assert "Bob: avg=95.0" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestStringFormattingPatterns:
    """Test string formatting propagation across cells."""

    def test_fstring_with_complex_expressions(self, nb_runner):
        """f-strings with complex expressions across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = {'name': 'Alice', 'scores': [90, 85, 92]}
            """),
                textwrap.dedent("""\
                avg = sum(data['scores']) / len(data['scores'])
                report = f"{data['name']}: avg={avg:.1f}, total={sum(data['scores'])}"
                print(report)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice: avg=89.0, total=267" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestStringProcessingPatterns:
    """Test complex string processing and regex patterns."""

    def test_format_spec_patterns(self, nb_runner):
        """Various format spec patterns."""
        nb_runner.create_notebook(
            [
                "value = 3.14159265",
                textwrap.dedent("""\
                results = [
                    f"{value:.2f}",
                    f"{value:.4e}",
                    f"{1000000:,}",
                    f"{0.75:.1%}",
                ]
                print(results)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "3.14" in output
        assert "1,000,000" in output
        assert "75.0%" in output


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringFormattingInteraction:
    """Test string formatting patterns with cache invalidation."""

    def test_format_method_edit(self, nb_runner):
        """Editing data used in str.format() should propagate."""
        nb_runner.create_notebook(
            [
                "name = 'Alice'\nage = 30",
                "template = '{name} is {age} years old'",
                "msg = template.format(name=name, age=age)",
                "print(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Alice is 30 years old" in out

        nb_runner.set_cell_source(1, "name = 'Bob'\nage = 25")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Bob is 25 years old" in out

    def test_string_template_edit(self, nb_runner):
        """Editing data used in string.Template should propagate."""
        nb_runner.create_notebook(
            [
                "from string import Template\nproduct = 'Widget'\nprice = 9.99",
                "t = Template('Buy $product for $$$price')",
                "msg = t.substitute(product=product, price=price)",
                "print(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Buy Widget for $9.99" in out

        nb_runner.set_cell_source(1, "from string import Template\nproduct = 'Gadget'\nprice = 19.99")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Buy Gadget for $19.99" in out

    def test_multiline_format_edit(self, nb_runner):
        """Editing data used in multiline formatting should propagate."""
        nb_runner.create_notebook(
            [
                "items = [('Apple', 3), ('Banana', 5)]",
                "lines = []\nfor name, qty in items:\n    lines.append(f'{name}: {qty}')",
                "report = '\\n'.join(lines)",
                "print(report)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Apple: 3" in out
        assert "Banana: 5" in out

        nb_runner.set_cell_source(1, "items = [('Cherry', 10), ('Date', 7), ('Fig', 2)]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Cherry: 10" in out
        assert "Date: 7" in out
        assert "Fig: 2" in out


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDebugPrintEdits:
    """Editing debug print patterns."""

    def test_edit_debug_format(self, nb_runner):
        """Edit the debug print format."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]  # debug print source",
                "print(f'len={len(data)} sum={sum(data)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3 sum=6" in nb_runner.get_output(2)

        # Change to more detailed format
        nb_runner.set_cell_source(2, "print(f'data={data} len={len(data)} min={min(data)} max={max(data)}')")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "data=[1, 2, 3]" in out
        assert "min=1" in out
        assert "max=3" in out

    def test_add_remove_debug_prints(self, nb_runner):
        """Edit output content between runs."""
        nb_runner.create_notebook(
            [
                "a = 5\nb = 10  # debug prints source",
                "c = a + b",
                "print(f'a={a} b={b} c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "a=5 b=10 c=15" in out

        # Change print format
        nb_runner.set_cell_source(3, "print(f'sum={c}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "sum=15" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFormatMapCustom:
    """Test str.format_map with custom mappings across cells."""

    def test_format_map_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: format_map with defaultdict
                "from collections import defaultdict\ndata = defaultdict(lambda: 'N/A', name='Alice', age=30)\nresult = '{name} is {age}, lives in {city}'.format_map(data)\nprint(f'result={result}')",
                # Cell 2: format with dict subclass
                "class SafeDict(dict):\n    def __missing__(self, key):\n        return f'<{key}>'\n\nsd = SafeDict(x=10, y=20)\nformatted = '{x} + {y} = {z}'.format_map(sd)\nprint(f'formatted={formatted}')",
                # Cell 3: complex formatting
                "template = 'Name: {name}, Score: {score:.1f}, Grade: {grade}'\nstudents = [\n    {'name': 'Alice', 'score': 95.5, 'grade': 'A'},\n    {'name': 'Bob', 'score': 82.3, 'grade': 'B'},\n]\nfor s in students:\n    print(template.format_map(s))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "result=Alice is 30, lives in N/A" in out1
        out2 = nb_runner.get_output(2)
        assert "formatted=10 + 20 = <z>" in out2
        out3 = nb_runner.get_output(3)
        assert "Name: Alice, Score: 95.5, Grade: A" in out3
        assert "Name: Bob, Score: 82.3, Grade: B" in out3

    def test_format_map_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = {'product': 'Widget', 'price': 9.99}\nmsg = 'Buy {product} for ${price:.2f}'.format_map(data)\nprint(f'msg={msg}')",
                "upper = msg.upper()\nprint(f'upper={upper}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Buy Widget for $9.99" in nb_runner.get_output(1)

        # Edit data
        nb_runner.set_cell_source(
            1,
            "data = {'product': 'Gadget', 'price': 19.99}\nmsg = 'Buy {product} for ${price:.2f}'.format_map(data)\nprint(f'msg={msg}')",
        )
        nb_runner.run_cells([1, 2])
        assert "msg=Buy Gadget for $19.99" in nb_runner.get_output(1)

    def test_format_map_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "info = {'name': 'test', 'version': '1.0'}\nheader = '{name} v{version}'.format_map(info)\nprint(f'header={header}')",
                "length = len(header)\nprint(f'length={length}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header=test v1.0" in nb_runner.get_output(1)
        # "test v1.0" = 9 chars
        assert "length=9" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=9" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringFormatMapTemplate:
    """string formatting with format_map and template."""

    def test_format_map_dict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import defaultdict",
                "template = '{name} is {age} years old from {city}'\ndata = defaultdict(lambda: 'N/A', name='Alice', age='30')\nresult = template.format_map(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Alice is 30 years old from N/A" in nb_runner.get_output(2)

    def test_string_template_safe(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from string import Template",
                "t = Template('$name owes $$${amount}')\nresult = t.safe_substitute(name='Bob')\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Bob owes $${amount}" in nb_runner.get_output(2)

    def test_format_map_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = {'x': 10, 'y': 20}",
                "result = '{x}+{y}'.format_map(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10+20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = {'x': 100, 'y': 200}")
        nb_runner.run_all()
        assert "result=100+200" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTemplateFormatMap:
    """Test string Template and format_map caching."""

    def test_template_substitute(self, nb_runner):
        """string.Template substitution with caching."""
        nb_runner.create_notebook(
            [
                "from string import Template",
                "tmpl = Template('Hello, $name! You are $age years old.')",
                "data = {'name': 'Alice', 'age': 30}",
                "result = tmpl.substitute(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Hello, Alice! You are 30 years old." in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "Hello, Alice!" in out2

    def test_template_edit_data(self, nb_runner):
        """Edit template data, verify output changes."""
        nb_runner.create_notebook(
            [
                "from string import Template",
                "tmpl = Template('$item costs $$${price}')",
                "data = {'item': 'Book', 'price': '25'}",
                "result = tmpl.substitute(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Book costs $25" in out

        nb_runner.set_cell_source(3, "data = {'item': 'Pen', 'price': '5'}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "Pen costs $5" in out2

    def test_format_map(self, nb_runner):
        """str.format_map with caching."""
        nb_runner.create_notebook(
            [
                "template = '{city} has {pop} people'",
                "data = {'city': 'NYC', 'pop': '8M'}",
                "result = template.format_map(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "NYC has 8M people" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "NYC has 8M people" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringTemplateSubstitute:
    """Test string Template substitution across cells."""

    def test_template_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic template
                "from string import Template\ntpl = Template('Hello $name, you are $age years old')\nresult = tpl.substitute(name='Alice', age=30)\nprint(f'result={result}')",
                # Cell 2: safe_substitute with missing key
                "tpl2 = Template('$greeting $name, welcome to $place')\nsafe = tpl2.safe_substitute(greeting='Hi', name='Bob')\nprint(f'safe={safe}')",
                # Cell 3: template from cell 1 data
                "report_tpl = Template('Report: $name is $age')\nreport = report_tpl.substitute(name='Alice', age=30)\nprint(f'report={report}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "result=Hello Alice, you are 30 years old" in out1
        out2 = nb_runner.get_output(2)
        assert "safe=Hi Bob, welcome to $place" in out2
        out3 = nb_runner.get_output(3)
        assert "report=Report: Alice is 30" in out3

    def test_template_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from string import Template\ntpl = Template('$item costs $$${price}')\nresult = tpl.substitute(item='Widget', price='9.99')\nprint(f'result={result}')",
                "msg = f'Buy now: {result}'\nprint(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Widget" in out1 and "9.99" in out1

        # Edit template
        nb_runner.set_cell_source(
            1,
            "from string import Template\ntpl = Template('$item on sale for $$${price}')\nresult = tpl.substitute(item='Gadget', price='4.99')\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        out2 = nb_runner.get_output(2)
        assert "Gadget" in out2 and "4.99" in out2

    def test_template_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from string import Template\ntpl = Template('$x + $y = $z')\neq = tpl.substitute(x='2', y='3', z='5')\nprint(f'eq={eq}')",
                "length = len(eq)\nprint(f'length={length}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "eq=2 + 3 = 5" in nb_runner.get_output(1)
        assert "length=9" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=9" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringTemplateEdits:
    """String template and formatting edit propagation."""

    def test_format_string_template(self, nb_runner):
        """Edit format template, output updates."""
        nb_runner.create_notebook(
            [
                "template = '{name} has {count} items'",
                "name = 'Alice'\ncount = 5",
                "msg = template.format(name=name, count=count)\nprint(f'msg = {msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = Alice has 5 items" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "template = 'User {name}: {count} items remaining'")
        nb_runner.run_all()
        assert "msg = User Alice: 5 items remaining" in nb_runner.get_output(3)

    def test_join_pattern_edit(self, nb_runner):
        """Edit separator in join operation."""
        nb_runner.create_notebook(
            [
                "sep = ', '",
                "words = ['hello', 'world', 'python']\nresult = sep.join(words)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hello, world, python" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "sep = ' | '")
        nb_runner.run_all()
        assert "result = hello | world | python" in nb_runner.get_output(2)

    def test_multiline_string_edit(self, nb_runner):
        """Edit multiline string template."""
        nb_runner.create_notebook(
            [
                "header = 'Report'\nfooter = 'End'",
                "body = 'Data: 42'",
                "doc = f'{header}\\n{body}\\n{footer}'\nprint(doc)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Report" in out
        assert "Data: 42" in out
        assert "End" in out

        nb_runner.set_cell_source(1, "header = '=== Summary ==='\nfooter = '=== Done ==='")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "=== Summary ===" in out2
        assert "=== Done ===" in out2

    def test_regex_sub_edit(self, nb_runner):
        """Edit regex pattern, substitution updates."""
        nb_runner.create_notebook(
            [
                "import re\npattern = r'\\d+'",
                "text = 'item1 and item22 plus item333'\nresult = re.sub(pattern, '#', text)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = item# and item# plus item#" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "import re\npattern = r'[a-z]+'")
        nb_runner.run_all()
        assert "result = #1 # #22 # #333" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestTemplatePatternEdits:
    """Editing string template/formatting patterns."""

    def test_edit_format_template(self, nb_runner):
        """Edit data used in string format template."""
        nb_runner.create_notebook(
            [
                "name = 'Alice'\nrole = 'engineer'",
                "msg = '{} is a {}'.format(name, role)\nprint(msg)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice is a engineer" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "name = 'Bob'\nrole = 'designer'")
        nb_runner.run_all()
        assert "Bob is a designer" in nb_runner.get_output(2)

    def test_edit_template_string(self, nb_runner):
        """Edit a Template string pattern."""
        nb_runner.create_notebook(
            [
                "from string import Template\ntmpl = Template('Hello, $name! You have $count messages.')",
                "result = tmpl.substitute(name='Alice', count=5)\nprint(result)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello, Alice! You have 5 messages." in nb_runner.get_output(2)

        # Change template
        nb_runner.set_cell_source(1, "from string import Template\ntmpl = Template('Hi $name, $count items in cart.')")
        nb_runner.run_all()
        assert "Hi Alice, 5 items in cart." in nb_runner.get_output(2)

    def test_edit_multiline_template(self, nb_runner):
        """Edit data used in multiline template."""
        nb_runner.create_notebook(
            [
                "items = [('apple', 2), ('banana', 3)]",
                "lines = []\nfor name, qty in items:\n    lines.append(f'{name}: {qty}')\noutput = ', '.join(lines)\nprint(output)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "apple: 2, banana: 3" in nb_runner.get_output(2)

        # Change items
        nb_runner.set_cell_source(1, "items = [('x', 10), ('y', 20), ('z', 30)]")
        nb_runner.run_all()
        assert "x: 10, y: 20, z: 30" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringAlignment:
    """string ljust/rjust/center and formatting alignment."""

    def test_ljust_rjust(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = [('apple', 3), ('banana', 12), ('cherry', 7)]",
                "lines = []\nfor name, qty in items:\n    lines.append(f'{name.ljust(10)}{str(qty).rjust(5)}')\nresult = '|'.join(lines)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple" in out
        assert "banana" in out

    def test_center_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "title = 'Hello'",
                "centered = title.center(20, '-')\nprint(f'centered={centered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "-------Hello--------" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "title = 'Hi'")
        nb_runner.run_all()
        assert "---------Hi---------" in nb_runner.get_output(2)

    def test_format_spec(self, nb_runner):
        nb_runner.create_notebook(
            [
                "values = [3.14159, 2.71828, 1.41421]",
                "formatted = [f'{v:.2f}' for v in values]\nprint(f'formatted={formatted}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted=['3.14', '2.72', '1.41']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringCenterLjustRjust:
    """Test string alignment methods across cells."""

    def test_alignment_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic alignment
                "text = 'Hi'\ncentered = text.center(10, '*')\nleft = text.ljust(10, '-')\nright = text.rjust(10, '.')\nprint(f'centered={centered}')\nprint(f'left={left}')\nprint(f'right={right}')",
                # Cell 2: use aligned strings
                "c_len = len(centered)\nl_len = len(left)\nr_len = len(right)\nall_same = c_len == l_len == r_len\nprint(f'all_len_10={all_same}')",
                # Cell 3: build table row
                "name = 'Item'.ljust(10)\nprice = '$9.99'.rjust(10)\nrow = name + '|' + price\nprint(f'row={row}')\nprint(f'row_len={len(row)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "centered=****Hi****" in out1
        assert "left=Hi--------" in out1
        assert "right=........Hi" in out1
        out2 = nb_runner.get_output(2)
        assert "all_len_10=True" in out2
        out3 = nb_runner.get_output(3)
        assert "row_len=21" in out3

    def test_alignment_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "word = 'OK'\naligned = word.center(8, '=')\nprint(f'aligned={aligned}')",
                "stripped = aligned.strip('=')\nprint(f'stripped={stripped}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "aligned====OK===" in nb_runner.get_output(1)
        assert "stripped=OK" in nb_runner.get_output(2)

        # Edit to use rjust
        nb_runner.set_cell_source(1, "word = 'OK'\naligned = word.rjust(8, '=')\nprint(f'aligned={aligned}')")
        nb_runner.run_cells([1, 2])
        assert "aligned=======OK" in nb_runner.get_output(1)
        assert "stripped=OK" in nb_runner.get_output(2)

    def test_alignment_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "label = 'Test'\npadded = label.center(12)\nprint(f'padded_len={len(padded)}')",
                "is_centered = padded.strip() == 'Test'\nprint(f'is_centered={is_centered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "padded_len=12" in nb_runner.get_output(1)
        assert "is_centered=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_centered=True" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringPaddingFormatting:
    """string ljust rjust center zfill formatting."""

    def test_ljust_rjust_center(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'hello'",
                "lj = text.ljust(10, '-')\nrj = text.rjust(10, '-')\nct = text.center(11, '*')\nprint(f'lj={lj} rj={rj} ct={ct}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lj=hello-----" in out
        assert "rj=-----hello" in out
        assert "ct=***hello***" in out

    def test_zfill(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = [1, 42, 100, 7]",
                "filled = [str(n).zfill(4) for n in nums]\nprint(f'filled={filled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filled=['0001', '0042', '0100', '0007']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringZfillNumFormat:
    """string zfill and numeric formatting."""

    def test_format_thousands(self, nb_runner):
        nb_runner.create_notebook(
            [
                "val = 1234567890",
                "formatted = f'{val:,}'\nprint(f'formatted={formatted}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted=1,234,567,890" in nb_runner.get_output(2)

    def test_zfill_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "code = '42'",
                "padded = code.zfill(6)\nprint(f'padded={padded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "padded=000042" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "code = '1234'")
        nb_runner.run_all()
        assert "padded=001234" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPprintFormatWidth:
    """Test pprint formatting across cells."""

    def test_pprint_width(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create nested structure
                "import pprint\ndata = {'alpha': [1, 2, 3], 'beta': [4, 5, 6], 'gamma': [7, 8, 9]}\nwide = pprint.pformat(data, width=120)\nnarrow = pprint.pformat(data, width=30)\nwide_lines = len(wide.splitlines())\nnarrow_lines = len(narrow.splitlines())\nprint(f'wide_lines={wide_lines}')\nprint(f'narrow_more={narrow_lines > wide_lines}')",
                # Cell 2: depth control
                "nested = {'a': {'b': {'c': {'d': 1}}}}\nshallow = pprint.pformat(nested, depth=2)\nhas_ellipsis = '...' in shallow\nprint(f'has_ellipsis={has_ellipsis}')",
                # Cell 3: compact mode
                "nums = list(range(15))\ncompact_str = pprint.pformat(nums, width=40, compact=True)\nnormal_str = pprint.pformat(nums, width=40, compact=False)\ncompact_lines = len(compact_str.splitlines())\nnormal_lines = len(normal_str.splitlines())\nprint(f'compact_fewer={compact_lines <= normal_lines}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "narrow_more=True" in out1
        out2 = nb_runner.get_output(2)
        assert "has_ellipsis=True" in out2
        out3 = nb_runner.get_output(3)
        assert "compact_fewer=True" in out3

    def test_pprint_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import pprint\nitems = {'x': 10, 'y': 20, 'z': 30}\nformatted = pprint.pformat(items, width=60)\nprint(f'has_x={\"x\" in formatted}')\nprint(f'has_z={\"z\" in formatted}')",
                "char_count = len(formatted)\nprint(f'chars={char_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1a = nb_runner.get_output(1)
        assert "has_x=True" in out1a

        # Add more items
        nb_runner.set_cell_source(
            1,
            "import pprint\nitems = {'x': 10, 'y': 20, 'z': 30, 'w': 40, 'v': 50}\nformatted = pprint.pformat(items, width=60)\nprint(f'has_x={\"x\" in formatted}')\nprint(f'has_v={\"v\" in formatted}')",
        )
        nb_runner.run_cells([1, 2])
        out1b = nb_runner.get_output(1)
        assert "has_v=True" in out1b

    def test_pprint_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import pprint\nobj = [{'key': i, 'val': i * 10} for i in range(5)]\npretty = pprint.pformat(obj, width=50)\nline_count = len(pretty.splitlines())\nprint(f'line_count={line_count}')",
                "has_key_3 = 'key' in pretty and '3' in pretty\nprint(f'has_key_3={has_key_3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "line_count=" in out1
        out2 = nb_runner.get_output(2)
        assert "has_key_3=True" in out2

        # Re-run - cache
        nb_runner.run_all()
        assert "has_key_3=True" in nb_runner.get_output(2)
