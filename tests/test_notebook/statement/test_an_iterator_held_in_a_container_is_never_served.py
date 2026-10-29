def test_a_constant_slice_draws_from_its_own_items_only():
    """``for k in big[:1]`` draws from the first item, not from all of
    ``big``: 30,000 other objects do not make it unreadable, and an
    iterator in the slice still counts."""
    import ast
    import types

    from cash.notebook.consumables import STREAM, drawn_stream_inputs_why

    big = [int] + [types.SimpleNamespace(i=i) for i in range(30_000)]
    tree = ast.parse("t = [k(3) for k in big[:1]][0]")
    assert drawn_stream_inputs_why(tree, {"big"}, {"t"}, {"big": big}) == {}
    held = [[iter(range(3))]] + big
    tree = ast.parse("t = [next(k[0]) for k in held[:1]]")
    assert drawn_stream_inputs_why(tree, {"held"}, {"t"}, {"held": held}) == {"held": STREAM}
    tree = ast.parse("t = [next(k) for k in held[0][:1]]")
    assert drawn_stream_inputs_why(tree, {"held"}, {"t"}, {"held": held}) == {"held": STREAM}
