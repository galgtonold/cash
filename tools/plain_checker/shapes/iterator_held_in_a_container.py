"""A draw from an iterator held in a list, a dict or an attribute.

The tail ``rest = list(h[0])`` re-run alone gives the exhausted ``[]`` in
plain Python. A hit (under floors0) handed back the Run All value and left
the iterator where it was, so the configs disagreed. ``iter`` is the control:
the iterator held by the name itself.
"""

from tools.plain_checker import notebook

_KINDS = {
    "iter": ("it = iter(range(6))", "next(it)", "list(it)"),
    "in-list": ("h = [iter(range(6))]", "next(h[0])", "list(h[0])"),
    "in-dict": ("h = {'k': iter(range(6))}", "next(h['k'])", "list(h['k'])"),
    "in-attr": ("class B:\n    pass\nb = B()\nb.it = iter(range(6))", "next(b.it)", "list(b.it)"),
}

SHAPES = [
    notebook(
        f"held-tail-{kind}",
        [src, f"first = [{draw} for _ in range(2)]\nprint(first)", f"rest = {tail}\nprint(rest)"],
        names=["first", "rest"],
        plan=["all", 2, 2, "restart", "all", 2],
    )
    for kind, (src, draw, tail) in _KINDS.items()
]
