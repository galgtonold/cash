"""An augmented assignment into a selection reads what selects it.

Round 25's r25s2 (BLOCKING): an upstream cell gained
``too_high = store_median > 50 * sales['price'].median()`` and
``sales.loc[too_high, ['price']] /= 100``; running a cell below raised
``UpstreamStateError: name 'too_high' is not defined``. A plain ``=`` into
``sales.loc[too_high, 'price']`` counted ``too_high`` as an input; ``/=`` did not,
so the rebuild ran the statement without its producer, and an edit to the mask
did not re-key it.
"""

import pytest

from cash.notebook.analysis import CodeAnalyzer


@pytest.mark.parametrize(
    "code, selector",
    [
        ("sales.loc[too_high, ['price']] /= 100", "too_high"),
        ("sales[mask] += 1", "mask"),
        ("counts[key] -= 1", "key"),
        ("a.b[k].c *= 2", "k"),
        ("grid[i][j] += 1", "j"),
    ],
)
def test_the_selector_is_an_input(code, selector):
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert selector in inputs
    assert selector not in outputs


def test_a_selector_defined_in_the_same_block_is_not_an_input():
    inputs, _ = CodeAnalyzer.analyze_code_block("k = 1\nd[k] += 1")
    assert "k" not in inputs
