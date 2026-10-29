"""A weak reference to a restored value held by a proxy, a closure or a
class registry."""

from tools.plain_checker import notebook

C0 = "import weakref, time\nclass Item:\n    def __init__(s, i):\n        s.id = i\nclass Bag(list):\n    pass\ntime.sleep(0.02)\nBIG = Bag(Item(i) for i in range(5))"
MUT = "BIG[0].id = -5"

SHAPES = [
    notebook(
        "weak-ref-holder-control",
        cells=[
            C0,
            "class Box:\n    def __init__(s, o):\n        s.h = weakref.ref(o)\n    def total(s):\n        return sum(i.id for i in s.h())\nbox = Box(BIG)",
            MUT,
            "t = box.total()  # @cash:no-cache",
        ],
        names=["t"],
    ),
    notebook(
        "weak-proxy-holder",
        cells=[
            C0,
            "class Box:\n    def __init__(s, o):\n        s.h = weakref.proxy(o)\n    def total(s):\n        return sum(i.id for i in s.h)\nbox = Box(BIG)",
            MUT,
            "t = box.total()  # @cash:no-cache",
        ],
        names=["t"],
    ),
    notebook(
        "weak-closure-holder",
        cells=[
            C0,
            "def watcher(o):\n    r = weakref.ref(o)\n    return lambda: sum(i.id for i in r())\ntot = watcher(BIG)",
            MUT,
            "t = tot()  # @cash:no-cache",
        ],
        names=["t"],
    ),
    notebook(
        "weak-class-registry-holder",
        cells=[
            C0,
            "class Reg:\n    refs = []\nReg.refs.append(weakref.ref(BIG))",
            MUT,
            "t = sum(i.id for i in Reg.refs[0]())  # @cash:no-cache",
        ],
        names=["t"],
    ),
]
