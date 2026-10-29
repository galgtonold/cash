"""A loop restored into an object while a closure, a default, a class, a
proxy or a variable points at a part of it."""

from tools.plain_checker import notebook

P = (
    "import time, weakref\nclass Child:\n    def __init__(self):\n        self.v = 0\n"
    "class P:\n    def __init__(self):\n        self.n = 0\n        self.child = Child()\n"
    "    def update(self, i=0):\n        self.n += 1\n        self.child.v = i\n"
)
LOOP = "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)"

SHAPES = [
    notebook(
        "loop-closure-weakref-to-part",
        cells=[
            P + "p = P()\ndef mk(c):\n    r = weakref.ref(c)\n    return lambda: r().v\nget = mk(p.child)",
            LOOP,
            "r = get()  # @cash:no-cache",
        ],
        names=["r"],
    ),
    notebook(
        "loop-closure-strong-to-part",
        cells=[
            P + "p = P()\ndef mk(c):\n    return lambda: c.v\nget = mk(p.child)",
            LOOP,
            "r = get()  # @cash:no-cache",
        ],
        names=["r"],
    ),
    notebook(
        "loop-default-arg-to-part",
        cells=[P + "p = P()\ndef look(c=p.child):\n    return c.v", LOOP, "r = look()  # @cash:no-cache"],
        names=["r"],
    ),
    notebook(
        "loop-class-attr-to-part",
        cells=[P + "p = P()\nclass Keep:\n    pass\nKeep.c = p.child", LOOP, "r = Keep.c.v  # @cash:no-cache"],
        names=["r"],
    ),
    notebook(
        "loop-proxy-to-part",
        cells=[P + "p = P()\npx = weakref.proxy(p.child)", LOOP, "r = px.v  # @cash:no-cache"],
        names=["r"],
    ),
    notebook(
        "loop-var-to-part",
        cells=[P + "p = P()\nc = p.child", LOOP, "r = (c.v, c is p.child)  # @cash:no-cache"],
        names=["r"],
    ),
]
