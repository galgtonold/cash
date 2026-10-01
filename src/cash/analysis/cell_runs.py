"""Which statements of a cell can be skipped or restored as a run.

Static analysis over a cell's top-level statements, for the cell executor:
:func:`written_later_in_cell` says which names a later statement writes
again (so a value is an intermediate of the cell), and :func:`jumpable_runs`
finds the runs of plain assignments a restore of a later version can jump.
"""

from __future__ import annotations

import ast

from .annotations import get_statement_annotations
from .code_analyzer import CodeAnalyzer

__all__ = ["jumpable_runs", "written_later_in_cell"]


def written_later_in_cell(body: list[ast.stmt]) -> list[frozenset[str]]:
    """For each top-level statement, the names a LATER statement of the cell
    writes -- rebinds or changes in place.

    A value every one of whose names is written again before the cell ends is
    an intermediate: the cell leaves a later version, and the end-of-cell pass
    writes that one to disk when restoring beats rebuilding
    (``TieredBackend.persist_from_memory``). Writing each intermediate to disk
    as it was made cost a cleaning cell 3.8 s of pickling on a cold run,
    for ~500 MB versions of ``sales`` that nothing restores.
    """
    outputs: list[set[str]] = []
    for node in body:
        try:
            _inputs, outs = CodeAnalyzer.analyze_code_block(ast.unparse(node))
        except Exception:  # noqa: BLE001 - unknown writes defer nothing
            outs = set()
        outputs.append(set(outs))
    later: list[frozenset[str]] = [frozenset()] * len(body)
    acc: set[str] = set()
    for i in range(len(body) - 1, -1, -1):
        later[i] = frozenset(acc)
        acc |= outputs[i]
    return later


def jumpable_runs(body: list[ast.stmt], raw_cell: str, touches_rng) -> dict[int, int]:
    """``{start: end}`` of the runs of plain assignments a restore can jump in.

    A run is consecutive top-level assignments that rebuild a name more than
    once (``sales = ...``, ``sales["t"] = ...``, ...), with no ``# @cash:``
    directive, no random draw, and no statement reading a name the run writes
    before the run has written it -- so where the run starts from is what the
    cell had before it, and every version inside it is the run's own. See
    ``UpstreamChecker.plan_cell_run``.
    """

    def plain(node) -> bool:
        if isinstance(node, ast.AnnAssign) and node.value is None:
            return False
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            return False
        if any(isinstance(n, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)) for n in ast.walk(node)):
            return False
        if "@cash:" in raw_cell and get_statement_annotations(raw_cell, node).has_directives():
            return False
        return not touches_rng(ast.unparse(node))

    runs: dict[int, int] = {}
    i, n = 0, len(body)
    while i < n:
        if not plain(body[i]):
            i += 1
            continue
        j = i
        while j < n and plain(body[j]):
            j += 1
        reads_writes = []
        for node in body[i:j]:
            try:
                inputs, outputs = CodeAnalyzer.analyze_code_block(ast.unparse(node))
            except Exception:  # noqa: BLE001 - an unanalysable statement ends the run
                reads_writes = []
                break
            reads_writes.append((set(inputs), set(outputs)))
        if len(reads_writes) >= 2:
            written = [w for _, w in reads_writes]
            everything = set().union(*written)
            rebuilt = any(sum(1 for w in written if name in w) > 1 for name in everything)
            so_far: set[str] = set()
            ordered = True
            for reads, writes in reads_writes:
                if reads & (everything - so_far):
                    ordered = False
                    break
                so_far |= writes
            if rebuilt and ordered and _writes_only_into_its_own_objects(body[i:j]):
                runs[i] = j
        i = j
    return runs


#: Calls known to return a new object that shares no data with their
#: arguments or receiver, unless told otherwise by a ``copy=`` argument.
#: Any other call -- a user helper, ``torch.from_numpy``, ``df.to_numpy()``,
#: ``np.asarray`` -- may hand back its input or a view of it.
_COPY_CALLS = frozenset(
    {
        # copies of the receiver or argument
        "copy",
        "deepcopy",
        "astype",
        "array",
        "list",
        "dict",
        "set",
        "sorted",
        # fresh arrays
        "zeros",
        "ones",
        "empty",
        "full",
        "zeros_like",
        "ones_like",
        "empty_like",
        "full_like",
        "arange",
        "linspace",
        # pandas methods that build a new frame
        "drop_duplicates",
        "dropna",
        "fillna",
        "drop",
        "merge",
        "concat",
        "assign",
        "sort_values",
        "sort_index",
        "reset_index",
    }
)

#: Indexers through which ``name.<indexer>[...] = v`` writes into ``name`` itself.
_INDEXERS = frozenset({"loc", "iloc", "at", "iat"})


def _makes_a_new_object(value: ast.expr) -> bool:
    """Whether *value* evaluates to an object no other name holds.

    Conservative: a name, an attribute, a slice (``arr[1:]`` is a view of
    ``arr``) or a column (``df['a']``) may be shared, and so may any call
    outside ``_COPY_CALLS`` or one passed ``copy=`` anything but ``True``
    (``a.astype(float, copy=False)`` can be ``a`` itself). A mask or a list of
    columns selects a copy; arithmetic, literals and the listed calls make a
    new object.
    """
    if isinstance(value, ast.Call):
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if not (name in _COPY_CALLS or name.startswith("read_")):
            return False
        return all(
            kw.arg != "copy" or (isinstance(kw.value, ast.Constant) and kw.value.value is True) for kw in value.keywords
        )
    if isinstance(
        value,
        (
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            ast.List,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            ast.JoinedStr,
        ),
    ):
        return True
    if isinstance(value, ast.Subscript):
        if isinstance(value.value, (ast.Tuple, ast.List)) and isinstance(value.slice, ast.Constant):
            items = value.value.elts
            index = value.slice.value
            return isinstance(index, int) and -len(items) <= index < len(items) and _makes_a_new_object(items[index])
        return isinstance(value.slice, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.List))
    return False


def _writes_into_the_name_itself(target: ast.expr) -> bool:
    """Whether a write to *target* lands in the object its base name holds.

    ``df['a'] = v``, ``obj.x = v`` and ``df.loc[m, 'a'] = v`` do. A deeper
    target -- ``y[0][1] = v``, ``obj.a.b = v`` -- writes into an object held
    *inside* it, which a shallow copy or a literal like ``[x]`` shares.
    """
    inner = target.value
    if isinstance(inner, ast.Name):
        return True
    return (
        isinstance(target, ast.Subscript)
        and isinstance(inner, ast.Attribute)
        and inner.attr in _INDEXERS
        and isinstance(inner.value, ast.Name)
    )


def _writes_only_into_its_own_objects(nodes: list[ast.stmt]) -> bool:
    """Whether every in-place write of the run lands in an object the run made.

    ``y = x; y[0] += 5`` changes ``x``, and ``v = arr[1:]; v += 1`` changes
    ``arr``: skipping or restoring those statements loses the change to the
    object outside the run. So a name the run writes into -- ``name[...] =``,
    ``name.attr =``, ``name += ...`` -- must have been bound in the run, before
    the write, by an expression that makes a new object, and the write must
    land in that object rather than in one it holds.
    """
    fresh: set[str] = set()
    for node in nodes:
        targets = [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else list(node.targets)
        for target in targets:
            if isinstance(node, ast.AugAssign) and isinstance(target, ast.Name):
                if target.id not in fresh:
                    return False
                continue
            base = target
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if base is not target:
                if not isinstance(base, ast.Name) or base.id not in fresh:
                    return False
                if not _writes_into_the_name_itself(target):
                    return False
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if len(targets) == 1 and names and _makes_a_new_object(node.value):
                fresh |= names
            else:
                # A rebinding by anything else (``a = b = x``, ``y, z = pair``)
                # may share; a write into ``name[...]`` keeps the name's object.
                fresh -= {
                    n.id
                    for t in targets
                    if not isinstance(t, (ast.Subscript, ast.Attribute))
                    for n in ast.walk(t)
                    if isinstance(n, ast.Name)
                }
    return True
