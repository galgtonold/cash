"""Which receivers a statement changes in place, before and after it runs.

A bare method call (``lst.append(x)``, ``model.fit(X)``) binds no name, so
AST analysis alone never reports its receiver as an output. The classifier
here decides, per receiver, whether it is known to mutate (route it into the
outputs and skip-cache the statement), can be observed by content after the
statement runs, or must be assumed to mutate.
"""

from __future__ import annotations

import ast
import hashlib
import pickle
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cash.analysis.cacheability_decision import receiver_is_identity_coupled
from cash.analysis.mutation_effects import (
    classify_receivers,
    comprehension_watch_candidates,
    drawn_on_arguments,
    is_module_name,
    live_function_source,
)
from cash.analysis.mutations import assigned_method_call_receivers, standalone_method_call_receivers
from cash.analysis.namespace_effects import (
    bare_call_arguments,
    call_arguments,
    fits_its_receiver,
    is_estimator,
    rewatch_exempt,
)
from cash.mutation_fingerprint import mutation_fingerprint
from cash.notebook.cache_key import statement_source_hash
from cash.notebook.call_effects import DigestHandoff
from cash.notebook.consumables import is_stream, watched_call_receivers
from cash.notebook.magic_effects import magic_call_arguments
from cash.notebook.restored_var import hashed_by_lineage
from cash.value_hash import budgeted_hash, compute_hash, is_costly

if TYPE_CHECKING:
    from cash.notebook._protocols import ShellProtocol
    from cash.notebook.tracking_state import TrackingState

__all__ = ["MutationClassifier"]

#: Methods that fit an estimator in place (`estimator_fit_receivers`).
_FIT_METHODS = frozenset({"fit", "partial_fit", "fit_transform", "fit_predict"})


class MutationClassifier:
    """Classifies a statement's method-call receivers and observes the ones
    it could not decide before execution."""

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        compute_hash: Callable[[Any], str] | None,
        digests: DigestHandoff | None = None,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.compute_hash = compute_hash
        #: The argument digests handed between this check and the calls
        #: inside the statement, so a big argument is hashed once before
        #: and once after (`DigestHandoff`).
        self.digests = digests if digests is not None else DigestHandoff()
        # Pre-execution fingerprints of a bare call's arguments, by statement
        # source hash -- see :meth:`classify`.
        self._arg_snapshots: dict[str, dict[str, str]] = {}
        # What a statement whose verdict is known could still change, by
        # source hash, and the keys it was seen running under -- see
        # :meth:`start_rewatch`.
        self._rewatch_candidates: dict[str, frozenset[str]] = {}
        self._rewatched_keys: dict[str, set[str]] = {}
        self._rewatch_snapshots: dict[str, dict[str, str | None]] = {}
        # Modules counted as changed only until a name handed to their
        # ``set_*`` call is seen changing, by source hash -- see :meth:`classify`.
        self._provisional: dict[str, dict[str, frozenset[str]]] = {}

    def observed_mutations(self, observe: set[str], source_hash: str) -> set[str]:
        """The receivers and bare-call arguments this statement changed.

        Compares each receiver in *observe* against its pre-statement content
        hash, and each argument :meth:`classify` fingerprinted against its
        fingerprint. Runs after execution, before capture, so a detected
        mutation joins the outputs.
        """
        mutated = {b for b in observe if self._receiver_mutated(b)}
        after = self.digests.digest_after
        for name, before in self._arg_snapshots.pop(source_hash, {}).items():
            if mutation_fingerprint(self.shell.user_ns.get(name), after) != before:
                mutated.add(name)
        return mutated

    def identity_coupled_call_receivers(self, tree: ast.Module | None) -> set[str]:
        """Receiver names in *tree* that are live matplotlib Figures/Axes.

        The narrow companion to :meth:`classify`, for the one
        case that must survive the control-body skip. A body statement carries an
        injected marker comment and its method-mutation classification is skipped
        wholesale, because bumping a receiver with a per-statement source the
        upstream simulation never reproduces desyncs the loop. That is right for
        ordinary receivers and wrong for a live Axes: ``ax.bar(...)`` in a loop
        body has NO outputs, so it is cached as an ordinary no-output call and
        restored as a no-op, while the sibling ``fig.savefig(...)`` still
        executes because it writes a file. The draw is skipped, the write is
        not, and the deliverable PNG is blank.

        Identity-coupling is the same single discriminator used to tell
        ``ax.hist()`` (draws on an Axes) from ``df.hist()`` (receiver-pure), and
        is used to widen scope to captured-return draws without
        over-invalidating. It imports no matplotlib and is False for everything
        that is not a Figure/Axes, so no loop that caches today stops caching.

        Callers use this to skip the CACHE only. It deliberately does NOT feed
        ``outputs``: the lineage bump is the part the control-body skip exists to
        prevent, and re-executing a draw needs none of it.
        """
        if tree is None:
            return set()
        receivers: set[str] = set()
        for base, _method in standalone_method_call_receivers(tree) | assigned_method_call_receivers(tree):
            value = self.shell.user_ns.get(base)
            if isinstance(value, types.ModuleType):
                continue  # ``plt.savefig()`` is a module call, not a receiver draw
            if receiver_is_identity_coupled(value):
                receivers.add(base)
        # An Axes handed to a plain function (``draw(ax, df)``) is drawn on
        # too -- the same ``drawn_args`` rule `classify`
        # applies outside a loop. In a loop body it was missed, and a re-run
        # saved every chart blank.
        return receivers | drawn_on_arguments(tree, self.shell.user_ns)

    def fitted_receivers(self, tree: ast.Module | None) -> set[str]:
        """Estimators a statement fits in place (``km.fit_predict(Z)``).

        The control-body companion of the ``fits_its_receiver`` routing in
        :meth:`classify`: a loop body skips that
        classification, so ``labels_k = km.fit_predict(Z)`` was served from the
        cache and ``models[k] = km`` kept an unfitted estimator.
        Cache-skip only, like :meth:`identity_coupled_call_receivers`.
        """
        if tree is None:
            return set()
        return {
            base
            for base, method in (standalone_method_call_receivers(tree) | assigned_method_call_receivers(tree))
            if fits_its_receiver(method, self.shell.user_ns.get(base))
        }

    def classify(
        self,
        tree: ast.Module | None,
        source_hash: str,
        outputs: set[str],
        extra_arguments: frozenset[str] = frozenset(),
        rewatch: bool = False,
    ) -> tuple[set[str], set[str], set[str], bool]:
        """Classify a statement's standalone method-call receivers.

        Returns ``(pre_route, observe, assumed, record_verdict)``:

        * ``pre_route`` — receivers to route into outputs + skip-cache now
          (statically known-mutating; a prior runtime verdict says it mutates;
          or assume-mutate because the receiver can't be reliably content-hashed
          — minus anything already in ``outputs``).
        * ``observe`` — tier-3 receivers to content-observe post-execution
          (verdict unknown, receiver reliably hashable).
        * ``assumed`` — tier-3 receivers assumed-mutating without observation
          (recorded into the verdict so the simulation reproduces them).
        * ``record_verdict`` — True when this statement's verdict is being learned.

        *extra_arguments* are watched as the arguments of a bare call are:
        what a statement of a loop body hands to code whose effect on it is
        unknown (``statement_watch_candidates``). With *rewatch*, what a
        known verdict does not hold is kept for :meth:`start_rewatch`.
        """
        verdict = self.tracking_state.mutation_verdicts.get(source_hash)
        arguments, kept_only = self._call_arguments(tree, outputs)
        arguments = arguments | (extra_arguments - outputs)
        classes = classify_receivers(tree, self.shell.user_ns, lambda: verdict, arguments=arguments)
        pre_route = set(classes.mutated)
        if classes.provisional:
            self._provisional[source_hash] = dict(classes.provisional)
        else:
            self._provisional.pop(source_hash, None)
        observe: set[str] = set()
        assumed: set[str] = set()
        # A receiver no rule or verdict decides is observed when it can be
        # hashed reliably, otherwise assumed to change (correctness first).
        snapshots: dict[str, str] = {}
        unwatched = True
        in_place = _in_place_receivers(tree) if classes.unknown_receivers else set()
        for base in classes.unknown_receivers:
            if self._receiver_observable(base):
                observe.add(base)
                continue
            # A frame, series or array is hashed by its lineage, not its
            # content: fingerprinted whole before and after instead, once per
            # text. Anything else (a random generator advanced by ``choice``)
            # is still assumed changed.
            # A method that changes its receiver in place when the data
            # calls for it (``arr.put(np.flatnonzero(arr < 0), 0)``) may
            # change nothing on the run that is looked at: assumed changed.
            if not _fingerprinted_receiver(self.shell.user_ns.get(base)) or base in in_place:
                assumed.add(base)
                pre_route.add(base)
                continue
            if unwatched:
                self.digests.watch(tree, self.shell.user_ns)
                unwatched = False
            fingerprint = mutation_fingerprint(self.shell.user_ns.get(base), self.digests.digest)
            if fingerprint is None:
                assumed.add(base)
                pre_route.add(base)
            else:
                snapshots[base] = fingerprint
        # An object handed to a bare call (`im.add_qc(df)`, `sc.tl.leiden(hv)`)
        # or to a function of the user's (`summary = add_features(df)`)
        # gets a full before/after fingerprint (`mutation_fingerprint`).
        # The result is learned into the verdict, so neither the next run nor
        # the simulation asks again: `print(df)` is learned as reading only.
        if classes.unknown_args and unwatched:
            self.digests.watch(tree, self.shell.user_ns)
        for name in classes.unknown_args:
            if not outputs and is_costly(self.shell.user_ns.get(name)):
                # Too costly to read (`budgeted_hash`: an openpyxl workbook,
                # 3 s a read, read twice), and handed on by a statement that
                # keeps nothing (`money_format(wb["P&L"])`), whose call is
                # there for what it does: assumed changed.
                assumed.add(name)
                pre_route.add(name)
                continue
            fingerprint = mutation_fingerprint(self.shell.user_ns.get(name), self.digests.digest)
            if fingerprint is None:
                if name in kept_only:
                    # `rows = fetch(conn)`: a handle that cannot be pickled
                    # is not assumed changed by a call whose result is kept,
                    # or every such statement would run every time.
                    continue
                assumed.add(name)
                pre_route.add(name)
            else:
                snapshots[name] = fingerprint
        if snapshots:
            self._arg_snapshots[source_hash] = snapshots
        record_verdict = verdict is None and bool(observe or assumed or snapshots)
        self._rewatch_candidates.pop(source_hash, None)
        if verdict is not None and rewatch:
            # Learned from the runs so far: one that changed nothing (an
            # empty comprehension, a copy of equal values) is no verdict on
            # other inputs (`start_rewatch`).
            fresh = classify_receivers(tree, self.shell.user_ns, lambda: None, arguments=arguments)
            unseen = (fresh.unknown_receivers | fresh.unknown_args) - set(verdict) - outputs - pre_route
            if unseen:
                user_ns = self.shell.user_ns
                unseen -= rewatch_exempt(tree, user_ns, lambda name: live_function_source(name, user_ns))
            if unseen:
                self._rewatch_candidates[source_hash] = frozenset(unseen)
        return pre_route - outputs, observe, assumed, record_verdict

    def provisional_modules(self, source_hash: str) -> dict[str, frozenset[str]]:
        """The modules :meth:`classify` counted as changed by a ``set_*``
        call handed names, with those names (`ReceiverClasses.provisional`)."""
        return self._provisional.pop(source_hash, {})

    def start_rewatch(self, source_hash: str, cache_key: str, tree: ast.Module | None = None) -> bool:
        """Before a statement whose verdict is known executes: fingerprint
        what it could change that the verdict does not hold, when it runs
        under a key not seen this session (new inputs). Its verdict was
        learned from earlier runs, and a run that changed nothing says
        nothing about this one. False when nothing is watched.

        *tree* is the statement: when it is one call, the digests taken here
        are the call's before it runs, and the call's after it are the
        statement's (`DigestHandoff`), as for a first run's fingerprints."""
        names = self._rewatch_candidates.pop(source_hash, None)
        if not names or not cache_key:
            return False
        seen = self._rewatched_keys.setdefault(source_hash, set())
        if cache_key in seen:
            return False
        if len(seen) >= 16:
            seen.clear()
        seen.add(cache_key)
        user_ns = self.shell.user_ns
        if tree is not None:
            self.digests.watch(tree, user_ns)
        self._rewatch_snapshots[source_hash] = {
            name: mutation_fingerprint(user_ns.get(name), self.digests.digest) for name in names
        }
        return True

    def rewatched_changes(self, source_hash: str) -> set[str]:
        """After the statement ran: what :meth:`start_rewatch` fingerprinted
        that changed (one that cannot be observed counts as changed)."""
        user_ns = self.shell.user_ns
        after = self.digests.digest_after
        return {
            name
            for name, before in self._rewatch_snapshots.pop(source_hash, {}).items()
            if before is None or name not in user_ns or mutation_fingerprint(user_ns[name], after) != before
        }

    def _call_arguments(self, tree: ast.Module | None, outputs: set[str]) -> tuple[set[str], set[str]]:
        """``(names, kept_only)``: the names to fingerprint around the
        statement -- what its calls are handed (``namespace_effects.call_arguments``)
        and the receivers of a method call whose result is bound
        (``consumables.watched_call_receivers``), both shared with the
        simulation -- and those among them only a call whose result is kept
        is handed (``r = work(st)``)."""
        user_ns = self.shell.user_ns
        watched = watched_call_receivers(tree, user_ns)
        handed = comprehension_watch_candidates(
            tree, user_ns, lambda name: is_module_name(name, user_ns), lambda name: live_function_source(name, user_ns)
        )
        handed = frozenset(name for name in handed if not is_stream(user_ns.get(name)))
        names = set(call_arguments(tree, user_ns) | watched | handed) - outputs
        return names, names - bare_call_arguments(tree, user_ns) - watched

    def magic_snapshots(self, node: ast.stmt, code: str) -> dict[str, str | None] | None:
        """Before the magic statement *node* (*code*) runs: the fingerprint
        of each name its ``%time``/``%timeit``/``%prun`` Python hands to a
        call (``magic_call_arguments``), for :meth:`note_magic_changes`.

        ``%time train(model)`` changes ``model`` as much as ``train(model)``
        does, and the plain statement is watched the same way. None when the
        statement's verdict is already known (learned once per text, as for a
        plain statement) or it hands nothing over. An argument of a bare call
        that cannot be fingerprinted is kept as None: it counts as changed.
        """
        if statement_source_hash(code) in self.tracking_state.mutation_verdicts:
            return None
        watched, bare = magic_call_arguments(node, self.shell.user_ns)
        snapshots: dict[str, str | None] = {}
        for name in watched:
            fingerprint = mutation_fingerprint(self.shell.user_ns.get(name), self.digests.fingerprints.digest)
            if fingerprint is None and name not in bare:
                continue
            snapshots[name] = fingerprint
        return snapshots or None

    def note_magic_changes(self, code: str, snapshots: dict[str, str | None] | None) -> set[str] | None:
        """After the magic statement *code* ran: the names it changed among
        those :meth:`magic_snapshots` fingerprinted, recorded as its verdict
        (``mutation_verdicts``, which the simulation reads back); None when
        nothing was watched."""
        if snapshots is None:
            return None
        user_ns = self.shell.user_ns
        changed = {
            name
            for name, before in snapshots.items()
            if before is None
            or name not in user_ns
            or mutation_fingerprint(user_ns[name], self.digests.fingerprints.digest) != before
        }
        self.tracking_state.mutation_verdicts[statement_source_hash(code)] = changed
        return changed

    def magic_changed_arguments(self, code: str) -> set[str]:
        """The names the magic statement *code* was seen changing through a
        call's argument (:meth:`note_magic_changes`)."""
        return set(self.tracking_state.mutation_verdicts.get(statement_source_hash(code), ()))

    def estimator_fit_receivers(
        self,
        tree: ast.Module | None,
        outputs: set[str],
    ) -> set[str]:
        """Receivers of a standalone ``est.fit(...)`` / ``est.partial_fit(...)``
        whose live value is a duck-typed sklearn estimator.

        A bare ``model.fit(X, y)`` mutates its receiver in place, so the general
        mutation classifier routes it to skip-caching. But a fit is the most
        expensive cell in an ML notebook, and a restart followed by Run All
        trained it again every time. Its cache key is already
        input-lineage-based (the estimator is an input), so it is cached and
        restored like an output. This narrow gate selects ONLY sklearn-style
        estimators (``is_estimator``) called through a fit method, so the
        estimator-caching path never loosens general mutation caching.

        A module is never an estimator (``pkg.fit(...)`` is a module
        function call). Names already surfaced as AST outputs are excluded too -- those are produced by
        an assignment (a fresh binding each run), already stored as outputs.
        """
        # The assignment form too: `X = vec.fit_transform(texts)` fits `vec`
        # as it returns X.
        candidates = standalone_method_call_receivers(tree) | assigned_method_call_receivers(tree)
        if not candidates:
            return set()
        receivers: set[str] = set()
        for base, method in candidates:
            if method not in _FIT_METHODS:
                continue
            if is_estimator(self.shell.user_ns.get(base)):
                receivers.add(base)
        return receivers - outputs

    def _receiver_observable(self, base: str) -> bool:
        """Return True if *base*'s value can be reliably content-hashed.

        A large object -- a frame, an array, a collection of more than 200
        items or one holding a frame (`hashed_by_lineage`) -- has its lineage
        as its session hash, not its content, so there is nothing to compare
        it with. Such receivers are excluded here and assume-mutated instead.
        """
        val = self.shell.user_ns.get(base)
        if val is None:
            return False
        return not hashed_by_lineage(val)

    def _receiver_mutated(self, base: str) -> bool:
        """Observe whether *base*'s content changed during this statement.

        Compares the post-execution content hash against the pre-statement hash
        in ``current_session_hashes``. Conservative (returns True) when the value
        is absent, has no prior recorded hash, or is unpicklable — in which case
        ``compute_hash`` returns an identity hash that can't reflect an in-place
        mutation.
        """
        val = self.shell.user_ns.get(base)
        if val is None or self.compute_hash is None:
            return True
        try:
            after = budgeted_hash(val, self.compute_hash)
        except (TypeError, ValueError, AttributeError, pickle.PicklingError):
            return True
        if after is None:
            return True  # too costly to read (`budgeted_hash`): assumed changed
        identity_hash = hashlib.sha256(str(id(val)).encode("utf-8")).hexdigest()
        if after == identity_hash:
            return True  # unpicklable -> identity hash -> mutation undetectable
        if self.compute_hash is compute_hash:
            self.digests.note_settled(val, after)
        before = self.tracking_state.current_session_hashes.get(base)
        return before is None or after != before


def _fingerprinted_receiver(value: object) -> bool:
    """A frame, series or array: an unlisted method called on one is looked
    at (`mutation_fingerprint`) rather than assumed to change it."""
    t = type(value)
    return ((t.__module__ or "").split(".", 1)[0], t.__name__) in {
        ("pandas", "DataFrame"),
        ("pandas", "Series"),
        ("numpy", "ndarray"),
    }


#: Methods of numpy arrays and pandas frames and series that change the
#: receiver in place, often only when its data calls for it (``sort`` of a
#: sorted array, ``update`` from an empty frame, ``put`` at no index): a run
#: that changed nothing says nothing about the next.
_IN_PLACE_METHODS = frozenset(
    {
        # numpy.ndarray
        "fill",
        "put",
        "sort",
        "partition",
        "resize",
        "itemset",
        "setfield",
        "setflags",
        "byteswap",
        # pandas DataFrame / Series
        "insert",
        "isetitem",
        "pop",
        "update",
        "_set_value",
        "__setitem__",
        "__delitem__",
        # in-place operators called by name
        "__iadd__",
        "__isub__",
        "__imul__",
        "__imatmul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__iand__",
        "__ior__",
        "__ixor__",
        "__ilshift__",
        "__irshift__",
    }
)


def _in_place_receivers(tree: ast.Module | None) -> set[str]:
    """The names *tree* calls an in-place method on (`_IN_PLACE_METHODS`, or
    any method handed ``inplace=`` other than a literal False), reached
    through attributes and subscripts."""
    found: set[str] = set()
    if tree is None:
        return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        in_place = method in _IN_PLACE_METHODS
        if not in_place:
            in_place = any(
                kw.arg == "inplace" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
                for kw in node.keywords
            )
        if not in_place:
            continue
        root: ast.AST = node.func.value
        while isinstance(root, (ast.Attribute, ast.Subscript, ast.Call)):
            root = root.func if isinstance(root, ast.Call) else root.value
        if isinstance(root, ast.Name):
            found.add(root.id)
    return found
