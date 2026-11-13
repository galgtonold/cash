"""`plot_runs._unseen_shown`: what a statement of a plot run showed past
cash's capture, as the kernel's display publisher reported it.

A ``with``/``if``/``try`` block runs as one unit whose displays reach the
notebook directly. The publisher also reports cash showing again what it
captured of a top-level statement; those are already the statement's own
and are not added twice. Where the publisher takes no hook, a block may
have shown anything, and the run holding it is not kept.
"""

from __future__ import annotations


def _unseen_shown(*args):
    from cash.notebook.ipython.plot_runs import _unseen_shown as unseen

    return unseen(*args)


PNG = {"data": {"image/png": "AAA", "text/plain": "<Figure>"}, "metadata": {}}
OTHER = {"data": {"image/png": "BBB", "text/plain": "<Figure>"}, "metadata": {}}


def test_a_display_shown_inside_a_block_is_added_to_its_last_part():
    parts = [("printed\n", "", []), ("", "", [])]
    assert _unseen_shown([("display_data", PNG)], parts, False, "if x:\n    plt.show()") is None
    assert parts[0][2] == [] and parts[1][2] == [PNG]


def test_cash_showing_a_captured_display_again_is_not_added_twice():
    parts = [("", "", [dict(PNG)])]
    assert _unseen_shown([("display_data", PNG)], parts, False, "plt.show()") is None
    assert parts[0][2] == [PNG]


def test_a_display_beside_the_captured_one_is_added():
    """Positive control for the one above: only the matching display is dropped."""
    parts = [("", "", [dict(PNG)])]
    assert _unseen_shown([("display_data", PNG), ("display_data", OTHER)], parts, False, "f()") is None
    assert parts[0][2] == [PNG, OTHER]


def test_a_display_with_an_id_cannot_be_kept():
    parts = [("", "", [])]
    assert _unseen_shown([("display_with_id", PNG)], parts, False, "if x:\n    show()") is not None


def test_a_display_that_is_not_plain_data_cannot_be_kept():
    parts = [("", "", [])]
    shown = {"data": {"application/x-widget": object()}, "metadata": {}}
    assert _unseen_shown([("display_data", shown)], parts, False, "if x:\n    show()") is not None


def test_unheard_a_block_is_refused_and_a_plain_statement_is_not():
    assert _unseen_shown([], [("", "", [])], True, "with ctx:\n    plt.show()") is not None
    assert _unseen_shown([], [("", "", [])], True, "plt.show()") is None
