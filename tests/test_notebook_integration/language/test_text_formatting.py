"""f-strings, format(), templates, textwrap and pprint across cells."""

import textwrap

import pytest


# Complex f-string interaction tests.
#
# Tests editing cells that contain complex f-string
# expressions and verifying proper output propagation.
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


# String template and formatting edit patterns.
#
# Tests various string formatting approaches with edits.
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


# Interaction test: string formatting with format_map and template patterns.
# Tests str.format_map, custom Mapping classes for format,
# and cross-cell string formatting pipelines.
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


# Interaction test: pprint formatting with width/depth control.
# Tests pprint.pformat with various width, depth, compact settings,
# and cross-cell pretty-printing of complex nested structures.
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


# Interaction test: string center, ljust, rjust padding operations.
# Tests fixed-width string formatting with different fill characters,
# cross-cell alignment pipelines, and cache invalidation on width changes.
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
class TestStringExpandtabs:
    """string expandtabs and whitespace handling."""

    def test_expandtabs(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'col1\\tcol2\\tcol3'",
                "expanded = text.expandtabs(8)\ncols = expanded.split()\nprint(f'cols={cols}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cols=['col1', 'col2', 'col3']" in nb_runner.get_output(2)

    def test_strip_variations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "s = '  hello  '",
                "l = s.lstrip()\nr = s.rstrip()\nb = s.strip()\nprint(f'l=[{l}] r=[{r}] b=[{b}]')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "l=[hello  ]" in out
        assert "r=[  hello]" in out
        assert "b=[hello]" in out

    def test_whitespace_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "raw = '  spaces  and\\ttabs  '",
                "cleaned = ' '.join(raw.split())\nprint(f'cleaned={cleaned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=spaces and tabs" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "raw = '\\t\\thello\\t\\tworld\\t\\t'")
        nb_runner.run_all()
        assert "cleaned=hello world" in nb_runner.get_output(2)


# String formatting and template interaction tests.
# Tests various string formatting patterns (f-strings, format(), Template)
# with cache invalidation when underlying data changes.
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


# Interaction test: string Template substitution.
# Tests string.Template with safe_substitute, missing keys,
# custom delimiters, and cross-cell template pipelines.
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


# string.Template and format_map patterns with caching.
# Tests Template substitution, safe_substitute, format_map, and edit propagation.
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


# String template/formatting interaction tests.
#
# Tests editing cells with various string formatting
# approaches and verifying correct output.
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
class TestTextwrapDedent:
    """textwrap, dedent, and multi-line string formatting."""

    def test_wrap_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nlong_text = 'The quick brown fox jumps over the lazy dog and then runs away'",
                "wrapped = textwrap.fill(long_text, width=30)\nlines = wrapped.count('\\n') + 1\nprint(f'lines={lines}')\nprint(f'wrapped={repr(wrapped)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lines=" in out

    def test_dedent_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = '    line1\\n    line2\\n    line3'",
                "dedented = textwrap.dedent(text)\nfirst = dedented.split('\\n')[0]\nprint(f'first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=line1" in nb_runner.get_output(2)
        # Edit text
        nb_runner.set_cell_source(1, "import textwrap\ntext = '        hello\\n        world'")
        nb_runner.run_all()
        assert "first=hello" in nb_runner.get_output(2)

    def test_indent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'line1\\nline2\\nline3'",
                "indented = textwrap.indent(text, '>>> ')\nprint(f'indented={repr(indented)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert ">>> line1" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapDedentFill:
    """textwrap module dedent and fill."""

    def test_dedent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nraw = '    hello\\n    world\\n    foo'",
                "cleaned = textwrap.dedent(raw)\nlines = cleaned.strip().split('\\n')\nprint(f'lines={lines}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines=['hello', 'world', 'foo']" in nb_runner.get_output(2)

    def test_fill(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'The quick brown fox jumps over the lazy dog and runs away'",
                "wrapped = textwrap.fill(text, width=20)\nline_count = len(wrapped.split('\\n'))\nprint(f'line_count={line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        count = int(nb_runner.get_output(2).split("line_count=")[1].strip())
        assert count >= 3

    def test_textwrap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'hello world test'",
                "shortened = textwrap.shorten(text, width=12, placeholder='...')\nprint(f'short={shortened}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "short=hello..." in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import textwrap\ntext = 'foo bar baz qux'")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "short=foo bar..." in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapDedentIndent:
    """textwrap dedent indent and fill wrapping."""

    def test_dedent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = '''\n        Hello World\n        This is indented\n        Three lines\n    '''\ndedented = textwrap.dedent(text).strip()\nlines = dedented.split('\\n')\nprint(f'lines={len(lines)} first={lines[0]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lines=3" in out
        assert "first=Hello World" in out

    def test_fill_width(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'The quick brown fox jumps over the lazy dog near the river'\nfilled = textwrap.fill(text, width=30)\nlines = filled.split('\\n')\nprint(f'line_count={len(lines)}')\nprint(f'max_len={max(len(l) for l in lines)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "max_len=" in out

    def test_wrap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'hello world foo bar'\nwrapped = textwrap.wrap(text, width=12)\nprint(f'parts={len(wrapped)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "text = 'a b c d e f g h'\nwrapped = textwrap.wrap(text, width=8)\nprint(f'parts={len(wrapped)}')"
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "parts=" in out2


# Interaction test: textwrap fill and shorten with break_on_hyphens.
# Tests textwrap.fill with break_long_words, break_on_hyphens,
# shorten with placeholder, and cross-cell text pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapFillBreak:
    """Test textwrap fill with break options across cells."""

    def test_textwrap_fill_break(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: fill with break options
                "import textwrap\ntext = 'This is a very-long-hyphenated-word-that-should-break and some more text after it'\nfilled_break = textwrap.fill(text, width=30, break_on_hyphens=True)\nfilled_no_break = textwrap.fill(text, width=30, break_on_hyphens=False)\nbreak_lines = len(filled_break.split('\\n'))\nno_break_lines = len(filled_no_break.split('\\n'))\nprint(f'break_lines={break_lines}')\nprint(f'no_break_lines={no_break_lines}')",
                # Cell 2: shorten
                "long = 'The quick brown fox jumps over the lazy dog near the river'\nshort = textwrap.shorten(long, width=30, placeholder='...')\nprint(f'shortened={short}')\nprint(f'short_len={len(short)}')",
                # Cell 3: combine
                "combo = textwrap.shorten(text, width=40, placeholder=' [...]')\nprint(f'combo={combo}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "break_lines=" in out1
        assert "no_break_lines=" in out1
        out2 = nb_runner.get_output(2)
        assert "shortened=" in out2
        assert len(nb_runner.get_output(2).split("shortened=")[1].split("\n")[0]) <= 30
        out3 = nb_runner.get_output(3)
        assert "combo=" in out3

    def test_textwrap_fill_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nparagraph = 'Python is a versatile language used for web development data science and automation'\nfilled = textwrap.fill(paragraph, width=25)\nline_count = len(filled.split('\\n'))\nprint(f'lines={line_count}')",
                "first = filled.split('\\n')[0]\nprint(f'first_line={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(1)
        assert "lines=" in out

        # Edit width
        nb_runner.set_cell_source(
            1,
            "import textwrap\nparagraph = 'Python is a versatile language used for web development data science and automation'\nfilled = textwrap.fill(paragraph, width=50)\nline_count = len(filled.split('\\n'))\nprint(f'lines={line_count}')",
        )
        nb_runner.run_cells([1, 2])
        out = nb_runner.get_output(2)
        assert "first_line=" in out

    def test_textwrap_fill_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nmsg = 'Hello World'\nresult = textwrap.shorten(msg, width=20, placeholder='...')\nprint(f'result={result}')",
                "is_truncated = '...' in result\nprint(f'truncated={is_truncated}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Hello World" in nb_runner.get_output(1)
        assert "truncated=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "truncated=False" in nb_runner.get_output(2)


# textwrap and string formatting patterns with caching.
# Tests textwrap.dedent, textwrap.fill, indent, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapFormat:
    """Test textwrap formatting operation caching."""

    def test_dedent_basic(self, nb_runner):
        """Dedent indented text, verify caching."""
        nb_runner.create_notebook(
            [
                "import textwrap",
                "raw = '    line1\\n    line2\\n    line3'",
                "cleaned = textwrap.dedent(raw)\nlines = cleaned.strip().split('\\n')",
                "print(f'count={len(lines)} first={lines[0]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=3" in out
        assert "first=line1" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=3" in out2

    def test_fill_wrap_edit(self, nb_runner):
        """textwrap.fill with width edit."""
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'The quick brown fox jumps over the lazy dog near the river bank'",
                "width = 20",
                "wrapped = textwrap.fill(text, width=width)\nline_count = len(wrapped.split('\\n'))",
                "print(f'lines={line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        lines_narrow = int(out.split("lines=")[1].strip())

        nb_runner.set_cell_source(3, "width = 40")
        nb_runner.run_all()
        out2 = nb_runner.get_output(5)
        lines_wide = int(out2.split("lines=")[1].strip())
        assert lines_wide < lines_narrow

    def test_indent_pattern(self, nb_runner):
        """textwrap.indent with prefix."""
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'line1\\nline2\\nline3'",
                "indented = textwrap.indent(text, '>>> ')\nfirst_line = indented.split('\\n')[0]",
                "print(f'first={first_line}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "first=>>> line1" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "first=>>> line1" in out2


# Interaction test: textwrap.wrap and shorten with custom settings.
# Tests textwrap.wrap with width, initial_indent, subsequent_indent,
# and textwrap.shorten across cells.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapWrapShorten:
    """Test textwrap.wrap and shorten across cells."""

    def test_wrap_shorten(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: wrap text
                "import textwrap\ntext = 'The quick brown fox jumps over the lazy dog and continues running through the forest'\nwrapped = textwrap.wrap(text, width=30)\nprint(f'lines={len(wrapped)}')\nfor line in wrapped:\n    print(f'  |{line}|')",
                # Cell 2: shorten
                "short = textwrap.shorten(text, width=40, placeholder='...')\nprint(f'short={short}')\nprint(f'short_len={len(short)}')",
                # Cell 3: indent
                "indented = textwrap.indent(text, prefix='>>> ')\nprint(f'indented={indented}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "lines=" in out1
        out2 = nb_runner.get_output(2)
        assert "..." in out2
        assert int(nb_runner.get_output(2).split("short_len=")[1].strip()) <= 40
        out3 = nb_runner.get_output(3)
        assert ">>> The quick" in out3

    def test_wrap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'Hello World this is a test of text wrapping'\nlines = textwrap.wrap(text, width=20)\nline_count = len(lines)\nprint(f'lines={line_count}')",
                "first = lines[0]\nprint(f'first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Change width
        nb_runner.set_cell_source(
            1,
            "import textwrap\ntext = 'Hello World this is a test of text wrapping'\nlines = textwrap.wrap(text, width=10)\nline_count = len(lines)\nprint(f'lines={line_count}')",
        )
        nb_runner.run_cells([1, 2])
        # Narrower width = more lines
        count = int(nb_runner.get_output(1).split("lines=")[1].strip())
        assert count > 3

    def test_wrap_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nresult = textwrap.fill('A short sentence for testing.', width=15)\nprint(f'filled={result}')",
                "line_count = result.count('\\n') + 1\nprint(f'count={line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "count=" in nb_runner.get_output(2)


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


# Complex f-strings & string formatting — cash caching with advanced formatting.
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
