"""A write to a plain list argument while its key is read, undone before the
body: another thread, played once (a file marks it done) by cash's own
steps, which plain Python never calls."""

from tools.plain_checker import decorator
from tools.plain_checker.shapes.from_import_corrected import _mod

_SRC = "def itotal(lst):\n    return sum(lst)\ndef rtotal(lst):\n    return sum(a for a, _b in lst)\n"

_PATCH = """
try:
    from cash.decorator import arg_hashing as _ah, purity_checks as _pc
except ImportError:
    _ah = None
ST = {'on': False}
if _ah is not None:
    _rpk, _rai = _ah.plain_key_part, _pc.PurityChecks.argument_identities
    def _pk(v):
        if ST['on'] and v is lst and 'old' not in ST:
            ST['old'] = get(lst); put(lst, OTHER)
            open('fired', 'w').close()
        return _rpk(v)
    def _ai(self, *a, **k):
        if 'old' in ST:
            put(lst, ST.pop('old')); ST['on'] = False
        return _rai(self, *a, **k)
    _ah.plain_key_part = _pk
    _pc.PurityChecks.argument_identities = _ai
"""


def _shape(name, func, build, get, put, other, change, source):
    return decorator(
        f"write-undone-while-keyed-{name}",
        setup=_mod("undone_mod", _SRC)
        + f"import os\nfrom undone_mod import *\nf = cache(undone_mod.{func})\nOTHER = {other}\n"
        + f"def get(l):\n    return {get}\ndef put(l, v):\n    {put}\nlst = {build}\n"
        + _PATCH,
        steps=[
            f"lst = {build}\nimport __main__\n__main__.lst = lst",
            "f(lst)",
            "f(lst)",
            "f(lst)",
            change,
            "ST['on'] = not os.path.exists('fired')",
            "f(lst)",
            "put(lst, OTHER)",
            "f(lst)",
        ],
        source=source,
    )


SHAPES = [
    _shape("ints", "itotal", "list(range(500))", "l[5]", "l[5] = v", "10 ** 6", "lst[7] = -1", "flat list"),
    _shape("many-ints", "itotal", "list(range(5000))", "l[5]", "l[5] = v", "10 ** 6", "lst[7] = -1", "flat list"),
    _shape(
        "rows",
        "rtotal",
        "[[i, i] for i in range(3000)]",
        "l[5][0]",
        "l[5][0] = v",
        "10 ** 6",
        "lst[7][0] = -1",
        "a row's item",
    ),
    _shape(
        "tuples",
        "rtotal",
        "[(i, i) for i in range(3000)]",
        "l[5]",
        "l[5] = v",
        "(10 ** 6, 5)",
        "lst[7] = (-1, 7)",
        "a row of a list of tuples",
    ),
]
