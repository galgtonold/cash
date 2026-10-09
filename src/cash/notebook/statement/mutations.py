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
from cash.analysis.mutation_effects import classify_receivers, drawn_on_arguments
from cash.analysis.mutations import assigned_method_call_receivers, standalone_method_call_receivers
from cash.analysis.namespace_effects import bare_call_arguments, call_arguments, fits_its_receiver, is_estimator
from cash.mutation_fingerprint import mutation_fingerprint
from cash.notebook.cache_key import statement_source_hash
from cash.notebook.call_effects import DigestHandoff
from cash.notebook.consumables import watched_call_receivers
from cash.notebook.magic_effects import magic_call_arguments
from cash.notebook.restored_var import hashed_by_lineage

if TYPE_CHECKING:
    from cash.notebook._protocols import ShellProtocol
    from cash.notebook.tracking_state import TrackingState

__all__ = ["MutationClassifier"]

#: Methods that fit an estimator in place, for `# @cash:cache-fit`.
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
        """
        verdict = self.tracking_state.mutation_verdicts.get(source_hash)
        arguments, kept_only = self._call_arguments(tree, outputs)
        classes = classify_receivers(tree, self.shell.user_ns, lambda: verdict, arguments=arguments)
        pre_route = set(classes.mutated)
        observe: set[str] = set()
        assumed: set[str] = set()
        # A receiver no rule or verdict decides is observed when it can be
        # hashed reliably, otherwise assumed to change (correctness first).
        for base in classes.unknown_receivers:
            if self._receiver_observable(base):
                observe.add(base)
            else:
                assumed.add(base)
                pre_route.add(base)
        # An object handed to a bare call (`im.add_qc(df)`, `sc.tl.leiden(hv)`)
        # or to a function of the user's (`summary = add_features(df)`)
        # gets a full before/after fingerprint (`mutation_fingerprint`).
        # The result is learned into the verdict, so neither the next run nor
        # the simulation asks again: `print(df)` is learned as reading only.
        snapshots: dict[str, str] = {}
        if classes.unknown_args:
            self.digests.watch(tree)
        for name in classes.unknown_args:
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
        return pre_route - outputs, observe, assumed, record_verdict

    def _call_arguments(self, tree: ast.Module | None, outputs: set[str]) -> tuple[set[str], set[str]]:
        """``(names, kept_only)``: the names to fingerprint around the
        statement -- what its calls are handed (``namespace_effects.call_arguments``)
        and the receivers of a method call whose result is bound
        (``consumables.watched_call_receivers``), both shared with the
        simulation -- and those among them only a call whose result is kept
        is handed (``r = work(st)``)."""
        user_ns = self.shell.user_ns
        watched = watched_call_receivers(tree, user_ns)
        names = set(call_arguments(tree, user_ns) | watched) - outputs
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

        Called ONLY for a statement carrying ``# @cash:cache-fit``; the
        default is to leave a bare fit on the skip-cache path, where it
        re-executes.  That re-execution does NOT by itself make aliases correct —
        ``backup = model`` breaks on its own restore, not on the fit.

        A bare ``model.fit(X, y)`` mutates its receiver in place, so the general
        mutation classifier routes it to skip-caching. But a fit is the most
        expensive cell in an ML notebook and its cache key is already
        input-lineage-based (the estimator is an input), so a user who asks for it
        can have it cached. This narrow gate selects ONLY sklearn-style
        estimators (``is_estimator``) called through a fit method, so the
        estimator-caching path never loosens general mutation caching.

        A module is never an estimator (``pkg.fit(...)`` is a module
        function call). Names already surfaced as AST outputs are excluded too -- those are produced by
        an assignment (a fresh binding each run), so an in-place transfer onto a
        pre-existing object would be wrong for them.
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

    def cache_fit_hint(self, receivers) -> str:
        """How to have a fitted estimator cached, when one of *receivers* is
        one: the refusal otherwise gives no way out."""
        for base in receivers:
            if is_estimator(self.shell.user_ns.get(base)):
                return (
                    f" -- `{base}` is an estimator being fitted; add `# @cash:cache-fit` "
                    "to cache the fit with it (see that directive's identity caveat)"
                )
        return ""

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
        if val is None:
            return True
        try:
            after = self.compute_hash(val)
        except (TypeError, ValueError, AttributeError, pickle.PicklingError):
            return True
        identity_hash = hashlib.sha256(str(id(val)).encode("utf-8")).hexdigest()
        if after == identity_hash:
            return True  # unpicklable -> identity hash -> mutation undetectable
        before = self.tracking_state.current_session_hashes.get(base)
        return before is None or after != before
