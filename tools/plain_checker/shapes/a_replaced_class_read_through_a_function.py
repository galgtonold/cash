"""An object of a class defined anew, read only through a notebook function.

Running the class cell again forgets each variable holding an object of the
old class, so the cell building it runs again. A cell that reads it only
through a function (``put(5)`` reaching ``boxes['s']``) must see what plain
Python gives, whether it runs in a Run All or on its own: on its own, the
object of the old class, as a direct read (``boxes['s'].hold(5)``) does.
"""

from tools.plain_checker import notebook

_CLASS = "class SBox:\n    held = []\n    def hold(self, d):\n        type(self).held.append(d)\n        return len(type(self).held)\n"
_BUILD = "boxes = {'s': SBox()}"
_FUNC = "def put(d):\n    return boxes['s'].hold(d), len(SBox.held)\n"
_FUNC_OBJ = "def put(d):\n    return boxes['s'].hold(d)\n"
_CALL = "import time\ntime.sleep(0.3)\nn = put(5)"

SHAPES = [
    notebook(
        "replaced-class-function-reader-run-all",
        [_CLASS, _BUILD, _FUNC, _CALL],
        names=["n"],
    ),
    notebook(
        "replaced-class-function-in-the-class-cell",
        [_CLASS + _FUNC, _BUILD, _CALL],
        names=["n"],
        plan=["all", "all", 0, 2, "restart", "all", 0, 2],
    ),
    notebook(
        "replaced-class-function-reads-only-the-object",
        [_CLASS, _BUILD, _FUNC_OBJ, _CALL],
        names=["n"],
        plan=["all", "all", 0, 3, "restart", "all", 0, 3],
    ),
    notebook(
        "replaced-class-read-directly",
        [_CLASS, _BUILD, "import time\ntime.sleep(0.3)\nn = boxes['s'].hold(5)"],
        names=["n"],
        plan=["all", "all", 0, 2, "restart", "all", 0, 2],
    ),
]
