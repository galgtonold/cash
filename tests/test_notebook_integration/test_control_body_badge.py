"""
Integration tests for control structure body statement display in badge
and spurious output fix.
"""
import pytest


def get_html_output(cell) -> str:
    """Extract HTML output from a cell's outputs."""
    html_parts = []
    for output in cell.get('outputs', []):
        if output.output_type in ('display_data', 'execute_result'):
            data = output.get('data', {})
            if 'text/html' in data:
                html_parts.append(data['text/html'])
    return '\n'.join(html_parts)


@pytest.mark.core
def test_if_else_badge_shows_body_statements(nb_runner):
    """
    Badge for an if/else block should contain the individual branch statements
    in an expandable section, not just the truncated header.
    """
    nb_runner.create_notebook([
        "x = 10",
        """\
if x > 5:
    y = x * 2
    label = 'big'
else:
    y = 0
    label = 'small'""",
    ])

    nb_runner.start_kernel()
    nb_runner.run_all()

    # Re-run to get a RESTORED badge
    nb_runner.run_cell(2)

    # Check that the HTML badge output contains body statements
    html = get_html_output(nb_runner.get_cell(2))
    # The badge should contain the body statements from the if block
    assert "y = x * 2" in html or "y = 0" in html or "label" in html, \
        f"Badge should show body statements, got: {html[:500]}"


@pytest.mark.core
def test_skipped_statement_no_spurious_output(nb_runner):
    """
    SKIPPED statements should not produce spurious text output.

    Previously, variable names like 'row_count' appeared as cell output
    because metrics['outputs'] stored variable names in the same key used
    for rich display outputs.
    """
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 2",
        "print(y)",
    ])

    nb_runner.start_kernel()
    nb_runner.run_all()

    # Run the same cells again — x=10 and y=x*2 should be SKIPPED
    nb_runner.run_cell(1)
    output1 = nb_runner.get_output(1)
    # x = 10 should not produce any text output (no print, no spurious var names)
    # It may be empty or only contain badge-related whitespace
    assert 'x' not in output1 or output1.strip() == '', \
        f"SKIPPED should not output variable names, got: {output1!r}"

    nb_runner.run_cell(2)
    output2 = nb_runner.get_output(2)
    assert 'y' not in output2 or output2.strip() == '', \
        f"SKIPPED should not output variable names, got: {output2!r}"


@pytest.mark.core
def test_try_except_badge_shows_body(nb_runner):
    """Badge for try/except shows body statements."""
    nb_runner.create_notebook([
        """\
try:
    result = 1 / 1
    status = 'ok'
except ZeroDivisionError:
    result = 0
    status = 'error'""",
    ])

    nb_runner.start_kernel()
    nb_runner.run_all()

    # Re-run for RESTORED badge
    nb_runner.run_cell(1)

    html = get_html_output(nb_runner.get_cell(1))
    # Should contain body statements
    assert "result" in html and "status" in html, \
        f"Badge should show body statements for try/except, got: {html[:500]}"


@pytest.mark.control
def test_while_badge_shows_body(nb_runner):
    """Badge for while loop shows body statements."""
    nb_runner.create_notebook([
        "i = 0",
        """\
while i < 3:
    i += 1""",
    ])

    nb_runner.start_kernel()
    nb_runner.run_all()

    # Re-run for RESTORED badge
    nb_runner.run_cell(2)

    html = get_html_output(nb_runner.get_cell(2))
    # Should contain the while condition and body
    assert "while" in html.lower() or "i += 1" in html, \
        f"Badge should show body statements for while, got: {html[:500]}"
