"""An id() of an object only a class or a closure holds."""

from tools.plain_checker import notebook

SLOW = "def slow(k):\n    time.sleep(0.05)\n    return {k: 'a'}\n"
SHAPES = [
    notebook(
        "id-of-object-on-class",
        cells=[
            "import time\nclass Thing:\n    pass\nclass Reg:\n    items = [Thing()]\n" + SLOW,
            "reg = slow(id(Reg.items[0]))",
            "r = reg.get(id(Reg.items[0]))  # @cash:no-cache",
        ],
        names=["r"],
    ),
    notebook(
        "id-of-object-in-closure",
        cells=[
            "import time\nclass Thing:\n    pass\ndef mk():\n    t = Thing()\n    return lambda: t\nget = mk()\n"
            + SLOW,
            "reg = slow(id(get()))",
            "r = reg.get(id(get()))  # @cash:no-cache",
        ],
        names=["r"],
    ),
]
