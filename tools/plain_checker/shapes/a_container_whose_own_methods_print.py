"""A container of a class of the user's whose own methods raise, or print
and count (from the batch 16 launch-bar review, LB2-2): cash reads it by the
builtin type's methods, so no cell shows a call plain Python never made.

Each shape: cell 0 defines p (p.child is the target) and the container class;
cell 1 binds a container holding id(p.child) (or a weakref to it, or nothing);
cell 2 reads it through the builtin type's methods (no user code) and prints
how often cash, or anyone, called the class's own methods.
"""

from tools.plain_checker import notebook

P = (
    "import time, weakref, collections, collections.abc\n"
    "CALLS = []\n"
    "class Q:\n    pass\n"
    "class P:\n    def __init__(self):\n        self.child = Q()\n"
    "p = P()\n"
)

METHODS = {
    "dict": [
        "__iter__",
        "__len__",
        "__getitem__",
        "items",
        "keys",
        "values",
        "__contains__",
        "get",
        "copy",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__eq__",
    ],
    "list": [
        "__iter__",
        "__len__",
        "__getitem__",
        "__contains__",
        "copy",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__eq__",
        "__reversed__",
    ],
    "set": ["__iter__", "__len__", "__contains__", "copy", "__reduce__", "__reduce_ex__", "__getstate__", "__eq__"],
}


def _cls(name, base, family, mode):
    body = []
    for m in METHODS[family]:
        if mode == "raise":
            act = f"raise RuntimeError('{m} raised')"
        else:
            act = f"CALLS.append('{m}'); print('CALLED {m}'); return {base}.{m}(self, *a, **k)"
        body.append(f"    def {m}(self, *a, **k):\n        {act}")
    return f"class {name}({base}):\n" + "\n".join(body) + "\n"


def shape(tag, base, family, mode, hold, make, read):
    cls = _cls("C", base, family, mode)
    return notebook(
        f"lb2-{tag}-{mode}-{hold}",
        cells=[P + cls, "time.sleep(0.05)\n" + make, f"print({read}, len(CALLS))  # @cash:no-cache"],
        names=[],
    )


SHAPES = []
for mode in ("raise", "count"):
    for hold in ("id", "weak", "none"):
        v = {"id": "id(p.child)", "weak": "weakref.ref(p.child)", "none": "7"}[hold]
        chk = {
            "id": "{} == id(p.child)",
            "weak": "{}() is p.child",
            "none": "{} == 7",
        }[hold]
        SHAPES += [
            shape(
                "dict",
                "dict",
                "dict",
                mode,
                hold,
                f"c = C()\ndict.__setitem__(c, 'k', {v})",
                chk.format("dict.__getitem__(c, 'k')"),
            ),
            shape(
                "odict",
                "collections.OrderedDict",
                "dict",
                mode,
                hold,
                f"c = C()\ncollections.OrderedDict.__setitem__(c, 'k', {v})",
                chk.format("dict.__getitem__(c, 'k')"),
            ),
            shape(
                "ddict",
                "collections.defaultdict",
                "dict",
                mode,
                hold,
                f"c = C(int)\ndict.__setitem__(c, 'k', {v})",
                chk.format("dict.__getitem__(c, 'k')"),
            ),
            shape(
                "list",
                "list",
                "list",
                mode,
                hold,
                f"c = C()\nlist.append(c, {v})",
                chk.format("list.__getitem__(c, 0)"),
            ),
            shape("set", "set", "set", mode, hold, f"c = C()\nset.add(c, {v})", chk.format("next(set.__iter__(c))")),
            shape(
                "userdict",
                "collections.UserDict",
                "dict",
                mode,
                hold,
                f"c = C()\nc.data['k'] = {v}",
                chk.format("c.data['k']"),
            ),
        ]
    # a mapping that is no subclass of a builtin container: its storage in an
    # attribute (a number, not an id: an id held there is the identity check's)
    SHAPES.append(
        notebook(
            f"lb2-mapping-{mode}-none",
            cells=[
                P + "class C(collections.abc.Mapping):\n"
                "    def __init__(self, d):\n        self._d = d\n"
                + "".join(
                    f"    def {m}(self, *a):\n        "
                    + (
                        "raise RuntimeError('x')"
                        if mode == "raise"
                        else f"CALLS.append('{m}'); print('CALLED {m}'); return getattr(self._d, '{m}')(*a)"
                    )
                    + "\n"
                    for m in ("__iter__", "__len__", "__getitem__", "items", "keys", "values")
                ),
                "time.sleep(0.05)\nc = C({'k': 7})",
                "print(c._d['k'] == 7, len(CALLS))  # @cash:no-cache",
            ],
            names=[],
        )
    )
