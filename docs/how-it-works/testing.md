# How cash is tested

A cache asks you for something unusual: permission to *not* run your code, and
trust that the answer it hands back instead is the one your code would have
produced. Every other kind of bug announces itself. A stale cache hit looks
exactly like a correct one.

So the question worth asking before you adopt this is not "does it work?" but
"how would they know if it didn't?" This page answers that, including the parts
where the answer is *we wouldn't, and here is what we do about it*.

Figures below are as of 0.5.0. The mechanisms matter more than the counts, and
they are the part that does not go stale.

## The shape of the suite

Roughly **8,800 tests** across three suites, spread over about 1,200 files:

| Suite | Size | What it covers |
|---|---|---|
| Unit | ~4,350 | The engine: cache keys, lineage, hashing, backends, invalidation |
| Notebook integration | ~4,150 | Real kernels executing real notebooks, cell by cell |
| Docs | ~330 | The documentation itself — see [below](#the-docs-are-tested-too) |

Every one runs on **15 platform combinations** — Python 3.10 through 3.14, on
Linux, Windows and macOS — on every push. The matrix is deliberately kept in
lockstep with the versions advertised on PyPI, because a version we claim to
support but never run is a support claim backed by nothing.

## The suite is a regression corpus, not a coverage target

Most of these tests are not there to exercise a feature. They are there because
something was once wrong, and this is the shape of the thing that was wrong.

That distinction changes what the suite is *for*. Coverage numbers measure which
lines ran; this measures which failures cannot come back. Tests are not pruned
for redundancy — two tests that look alike may pin two genuinely different
regressions, and the cost of keeping both is a few milliseconds while the cost of
deleting the wrong one is a bug that returns silently.

`xfail_strict` is on, which means a test marked "expected to fail" that starts
passing **fails the build**. A limitation that quietly gets fixed cannot sit in
the suite misreporting itself.

## Tests must prove they can fail

A test that passes against unfixed source is not evidence. It is a green
checkmark with nothing behind it.

[`scripts/fails_first.py`](https://github.com/galgtonold/cash/blob/main/scripts/fails_first.py)
exists to catch that: it stashes the fix, runs the new test against the
*unfixed* source, restores, and reports which tests failed. A test that passes
there is vacuously green and gets rewritten.

Its docstring enumerates four ways a test can be vacuously green — **all four of
which have actually shipped in this repository**:

1. **The mechanism never engages.** A decorator test whose function is faster
   than the persistence floor never writes to disk, so the assertion holds
   whether or not the bug exists.
2. **Empty input trivially satisfies the assertion.** "Is this output
   encodable?" passes for an empty string — so a harness that executed nothing
   looks like a pass.
3. **A different gate is substituted for the real one.** `mkdocs build --strict`
   validates links and never executes a Python fence, so docs whose code raises
   `NameError` still build green.
4. **State is checked instead of behaviour.** Asserting a policy object exists
   passes even when nothing ever calls it.

Publishing that list is the point. Every project has vacuously green tests; most
have never gone looking.

## The docs are tested too

Documentation drifts from code silently, and a caching library's docs are load
bearing — if the page says a change invalidates and it doesn't, the reader is
about to trust a stale value.

Two mechanisms, both blocking:

**Python fences are executed.** The docs suite auto-discovers every `.md` file
under `docs/`, `examples/` and the repository root, and runs its Python fences
through a real harness. There is no whitelist to maintain, so a new page is
covered the day it lands. Sample output in the docs is checked against what the
code actually prints.

**Prose is pinned to source.** Around **164 claims** across the documentation carry an
anchor naming the function that decides them, plus a fingerprint of that
function's normalized source:

```markdown
<!-- claim: cash/core.py:Cash.cache @b3cd263b -->
```

When the code changes, the fingerprint moves and the claim surfaces in a
re-verification queue. Clearing that queue is a **release gate** — the publish
workflow re-runs it with drift promoted from advisory to blocking, so a release
cannot go out resting on prose nobody re-read.

The tooling refuses to make this easy to fake: re-pinning without reading is
called out in its own help text as manufacturing the appearance of verification.
In practice most drift is a moved fingerprint on unchanged behaviour — but not
all of it. Preparing 0.4.1 surfaced three drifted claims about `%cash_stats`, and
**two of them had genuinely become wrong**: both enumerated what the command
prints and had gone stale the moment a line was added.

## Independent adversarial rounds

Before a release that widens the audience, cash goes through a round of
independent testing: people who did not write it, given real workloads, trying to
make it serve a wrong answer. A round must come back clean before that kind of
release ships.

This has been the most productive single source of real bugs — roughly one
correctness defect per five testers, in rounds where the automated suite was
entirely green. The findings are adjudicated rather than accepted: each is
reproduced independently before it is believed, because roughly a third of
reported issues turn out to be stale, environmental, or a misreading of correct
behaviour.

## Packaging is gated separately

The test suite is **structurally blind** to packaging bugs. It runs against an
editable install with every optional dependency present — which is not what a
user gets.

So the wheel is gated on its own terms: built, checked, and installed into a bare
virtual environment with **no optional dependencies at all**, from both the wheel
and a rebuild from the sdist, then exercised. A separate harness drives a real
Jupyter server against the built wheel, because driving cells through a test
client is not the same as a kernel a user would start.

## What this does *not* catch

This is the section that matters, and the reason the rest is worth believing.

**Silent degradation is the hard class.** When cash catches a problem, handles
it, and recomputes instead, everything stays green — because recomputing *is*
correct. The result is right. It just cost you the thing you installed cash for.

That is not hypothetical. In 0.4.1 we fixed a bug where **every Windows run was
silently discarding cache writes**. Windows refuses to replace a file while any
handle has it open; cash expects concurrent readers by design and writes on a
background thread, so the collision was routine. Each occurrence threw the entry
away and recomputed the work.

It survived every one of the ~8,800 tests, on every one of the 15 platform
combinations, through multiple adversarial rounds. Nothing raised. Nothing went
red. It was found by reading the full logs of jobs that had *passed*.

Two things came out of that, both shipped in 0.4.1:

- Discarded writes are now recorded when they happen and reported by
  [`%cash_stats`](../magics.md#cash_stats), so the failure is visible to you
  rather than inferred from savings that never arrive.
- **CI now fails if any cache write was discarded during a test run.** The class
  can no longer pass silently.

**Other things we know we do not catch.** Prose drift in tables and narrative
text — only Python fences and pinned claims are checked, so a wrong sentence
about behaviour is caught by a human or not at all. Timing-dependent behaviour on
hardware unlike CI's. And the long tail of interactions between a real user's
libraries that no fixture anticipates.

The [known limitations](../known-limitations.md) page is where the specific ones
live. It runs to several hundred lines, deliberately — it is easier to trust a
tool that tells you where it breaks than one that implies it never does.

## Reproducing any of this

Nothing above needs special access:

```bash
pytest tests/ --ignore=tests/test_notebook_integration --ignore=tests/test_wheel_gate --ignore=tests/docs
pytest tests/test_notebook_integration
pytest tests/docs
python scripts/claims.py --queue
python scripts/fails_first.py <your new test>
```

The CI configuration is
[in the repository](https://github.com/galgtonold/cash/blob/main/.github/workflows/ci.yml),
and every run's full logs are public.
