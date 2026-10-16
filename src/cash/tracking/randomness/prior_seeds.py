"""Seeds set before cash watched the ``seed`` functions.

cash learns that a global stream is seeded by wrapping ``random.seed`` and
``numpy.random.seed`` (`lineage.watch_seeds`). A seed set before that wrap
ran unobserved, and the stream was then taken for one seeded from fresh
entropy: a draw was keyed as unseeded, so ``np.random.seed(2)`` was served
the value drawn under ``np.random.seed(1)``.

The wrap starts at ``import cash``, and a stream module imported later is
wrapped as it loads (`RngImportHook`), so only a stream already loaded when
cash is imported can carry a seed cash did not see. For such a stream,
`seeded_before_watch` judges what happened before:

* numpy's global ``RandomState`` keeps the ``SeedSequence`` it drew its
  entropy from at import, and any ``np.random.seed(...)`` drops it. No seed
  sequence means a seed ran.
* Python's ``random`` keeps no such trace, so the code that ran before the
  import is read: the importing frames' own lines above the current one
  (a script's ``random.seed(0)`` above ``import cash``), and in IPython the
  cells run before the current one.

Every doubt resolves to "seeded": a seeded stream is keyed on where it
stands, which costs a recompute when it was in fact unseeded, never a
wrong value.
"""

from __future__ import annotations

import importlib.abc
import linecache
import os
import sys
import sysconfig
import textwrap
from types import ModuleType

from .detect import get_value_seeding_rng_modules

#: The modules whose global stream cash keys, and their import names.
WATCHED_STREAM_MODULES = ("random", "numpy.random", "torch")


def _library_roots() -> tuple[str, ...]:
    roots: set[str] = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            path = sysconfig.get_paths().get(key)
        except Exception:  # noqa: BLE001 - an odd install only widens the scan
            path = None
        if path:
            roots.add(os.path.realpath(path))
    return tuple(sorted(roots))


_CASH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def _scan(code: str) -> set[str]:
    try:
        return get_value_seeding_rng_modules(code)
    except (SyntaxError, ValueError, RecursionError):
        return set()


def _frame_prefix_seeds(frame) -> set[str]:
    """Modules seeded by *frame*'s own code above the line it stands on."""
    filename = frame.f_code.co_filename
    lines = linecache.getlines(filename, frame.f_globals)
    if not lines:
        return set()
    stop = max(frame.f_lineno - 1, 0)
    start = max(frame.f_code.co_firstlineno - 1, 0) if frame.f_code.co_name != "<module>" else 0
    prefix = "".join(lines[start:stop])
    try:
        return get_value_seeding_rng_modules(textwrap.dedent(prefix))
    except (SyntaxError, ValueError, RecursionError):
        # The line stands inside a block, so the lines above it are not a
        # whole program: read the whole file. A seed below the line is then
        # taken as one above it, which only keys a stream that was not seeded.
        return _scan("".join(lines))


def _user_frames():
    roots = _library_roots()
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        real = os.path.realpath(filename) if not filename.startswith("<") else filename
        if not (
            filename.startswith("<frozen")
            or real.startswith(_CASH_ROOT + os.sep)
            or any(real.startswith(root + os.sep) for root in roots)
        ):
            yield frame
        frame = frame.f_back


def _ipython_history_seeds() -> set[str]:
    """Modules seeded with a value by a cell run before the current one."""
    ipython = sys.modules.get("IPython")
    get = getattr(ipython, "get_ipython", None) if ipython is not None else None
    shell = get() if callable(get) else None
    namespace = getattr(shell, "user_ns", None)
    cells = namespace.get("In") if isinstance(namespace, dict) else None
    if not isinstance(cells, list):
        return set()
    found: set[str] = set()
    for cell in cells[:-1]:
        if isinstance(cell, str) and "seed" in cell:
            found |= _scan(cell)
    return found


def _source_seeds() -> set[str]:
    found = _ipython_history_seeds()
    for frame in _user_frames():
        found |= _frame_prefix_seeds(frame)
    return found


def _numpy_seed_sequence_dropped(np_random: ModuleType) -> bool | None:
    """Whether numpy's global stream lost its import-time seed sequence,
    which every ``np.random.seed(...)`` drops; None when it cannot be read."""
    try:
        bit_generator = np_random.mtrand._rand._bit_generator
    except AttributeError:
        return None
    if not hasattr(bit_generator, "_seed_seq"):
        return None
    return bit_generator._seed_seq is None


def seeded_before_watch(modules: set[str]) -> set[str]:
    """The streams of *modules*, each loaded before cash watched its seed
    function, whose state may come from a seed set in that time."""
    seeded: set[str] = set()
    unknown: set[str] = set()
    for module in modules:
        if module == "numpy.random":
            np_random = sys.modules.get("numpy.random")
            dropped = _numpy_seed_sequence_dropped(np_random) if np_random is not None else False
            if dropped:
                seeded.add(module)
            elif dropped is None:
                unknown.add(module)
        elif module == "random":
            unknown.add(module)
    if unknown:
        try:
            seeded |= _source_seeds() & unknown
        except Exception:  # noqa: BLE001 - unreadable: take the stream as seeded
            seeded |= unknown
    return seeded


class _WatchingLoader(importlib.abc.Loader):
    """Runs the module's own loader, then wraps its ``seed``."""

    def __init__(self, loader, on_loaded) -> None:
        self._loader = loader
        self._on_loaded = on_loaded

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module) -> None:
        # Leave the module as its own loader would: nothing reads this one.
        module.__loader__ = self._loader
        if getattr(module, "__spec__", None) is not None:
            module.__spec__.loader = self._loader
        self._loader.exec_module(module)
        try:
            self._on_loaded(module.__name__)
        except Exception:  # noqa: BLE001 - a failed wrap only loses the watch
            pass

    def __getattr__(self, name):
        return getattr(self._loader, name)


class RngImportHook(importlib.abc.MetaPathFinder):
    """Wraps the ``seed`` of a stream module imported after cash, as it
    loads: before user code can seed it unobserved."""

    def __init__(self, on_loaded) -> None:
        self._on_loaded = on_loaded
        self._finding: set[str] = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in WATCHED_STREAM_MODULES or fullname in self._finding:
            return None
        self._finding.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._finding.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _WatchingLoader(spec.loader, self._on_loaded)
        return spec


__all__ = ["WATCHED_STREAM_MODULES", "RngImportHook", "seeded_before_watch"]
