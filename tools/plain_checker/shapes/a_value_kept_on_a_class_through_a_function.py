"""A value kept on a class through a function: a staticmethod closure, a
default, a partial or a closure in a list on the class. An edit to the bump
must reach the read."""

from tools.plain_checker import notebook

DATA = "import time\ndata = (time.sleep(0.2), {'rows': [{'v': i} for i in range(3000)], 'meta': {'n': 1}})[1]"
READ = "result = (time.sleep(0.2), sum(r['v'] for r in data['rows']) + data['meta']['n'])[1]"
BUMP = "    @classmethod\n    def bump(cls, n):\n        cls.getter()['n'] += n"
VARIANTS = {
    "closure-on-class": "        type(self).getter = staticmethod(lambda: d)",
    "default-on-class": "        def g(x=d):\n            return x\n        type(self).getter = staticmethod(g)",
    "partial-on-class": "        type(self).getter = functools.partial(lambda x: x, d)",
    "closure-in-list-on-class": "        type(self).fs.append(lambda: d)\n        type(self).getter = staticmethod(lambda: type(self).fs[-1]())",
}
SHAPES = []
for name, hold in VARIANTS.items():
    cls = f"import functools\nclass SBox:\n    getter = None\n    fs = []\n    def hold(self, d):\n{hold}\n{BUMP}"
    SHAPES.append(
        notebook(
            name,
            cells=[cls, "sbox = SBox()", DATA, "sbox.hold(data['meta'])", "SBox.bump(5)", READ],
            names=["result"],
            edits={1: {4: "SBox.bump(7)"}},
        )
    )
