# Purity markers

!!! info "Applies to: decorator"
    Scripts, services and libraries that use `@cash.cache` and want to tell
    cash something about a helper or a class.

On the first call of a cached function, cash reads its code and its helpers and
warns about side effects a cache hit would skip
([Side effects](../../decorator.md#side-effects)). Three markers let you correct
that reading where you know better:

| Marker | Put it on | Effect on a cached function that uses it |
|---|---|---|
| `@cash.pure` | a helper with no side effects | No warning about the helper |
| `@cash.stateful` | a helper whose side effect matters | Warns about every call to it; `strict=True` raises |
| `cash.opaque` | a class passed as an argument | Its code is left out of the key |

<!-- claim: cash/purity_analyzer.py:PurityAnalyzer._analyze_uncached @8dc957f9 -->
`@cash.pure` and `@cash.stateful` change what cash reports, not what it keys. A
marked helper's code is part of the key of every cached function that calls it,
as an unmarked helper's is, so editing it recomputes them.

To accept one side effect in one place, you don't need a marker: put
`# @cash:assume-safe` on the line (see [Side effects](../../decorator.md#side-effects)).

In a notebook, `@cash.stateful` has a stronger meaning (a statement that calls
it is never cached); see the [Notebook guide](../../notebook_caching_api.md#what-gets-cached).

## `@cash.pure`: trust this helper

<!-- claim: cash/purity.py:pure @f53a99f5, cash/purity_analyzer.py:PurityAnalyzer.analyze @7dc29215 -->
Mark a helper `@cash.pure` when its result depends only on its arguments and it
has no effect you care about: no writes, no network, no in-place change to its
arguments. Cash then stops reporting it:

```python
import cash

@cash.pure
def normalise(values):
    total = sum(values)
    return [v / total for v in values]

@cash.cache
def shares(values):
    return normalise(values)

shares((1, 2, 5))   # first call: computes
shares((1, 2, 5))   # cache hit
```

!!! warning "Cash does not check a `@pure` helper"
    Cash takes the marker at its word: it reports nothing about the helper or
    the functions it calls, even an effect it would otherwise warn about. For a
    helper in your own project you rarely need `@pure`: cash already reads it
    and reports only real findings.

The marker earns its keep on **library** functions that cash reports but you
have checked. Call it on the function; it marks the function itself:

```python
import cash
import pandas as pd

cash.pure(pd.DataFrame.merge)   # checked: merge returns a new frame
```

A built-in or C function can't take the marker, so this raises
`AttributeError` instead of doing nothing.

## `@cash.stateful`: this helper has an effect that matters

<!-- claim: cash/purity.py:stateful @f86f4e92 -->
Mark a helper `@cash.stateful` when calling it does something a cache hit must
not skip silently: it posts a notification, writes to a database, updates a
model registry. A cached function that calls it warns
[`IMPURE-SIDE-EFFECTS`](../../warnings.md#impure-side-effects) on its first
call, naming the helper. It still caches; `strict=True` makes the finding an
error:

<!-- test:expect-warning reason="the IMPURE-SIDE-EFFECTS warning for a @stateful callee is the point of this example" -->
```python
import cash
import requests

@cash.stateful
def notify(message):
    requests.post("https://hooks.example.com/runs", json={"text": message})

@cash.cache
def nightly_report(day):
    notify(f"report for {day} built")
    return {"day": day, "rows": 1000}

nightly_report("2026-09-10")   # first call: warns about notify(), then caches
```

Here the fix is to move `notify` out of the cached function, so it runs on
every call and the report is still cached.

Mark a library function the same way, from outside:

```python
import cash
import pandas as pd

cash.stateful(pd.DataFrame.to_sql)   # writes to a database
```

## `cash.opaque`: leave a class out of the key

<!-- claim: cash/__init__.py:opaque @679c15ff, cash/decorator/arg_hashing.py:is_opaque @c98ecac3 -->
A class or function of yours passed as an argument is keyed by its code, so an
edit to it recomputes the call. For a class whose code does not affect the
result (a marker type, a vendored class that changes for unrelated reasons),
that costs recomputes for nothing. Mark it opaque:

```python
import cash

@cash.opaque
class RenderTarget:
    def __init__(self, name):
        self.name = name

VendorWidget = type("VendorWidget", (), {})   # stands in for a library class
cash.opaque(VendorWidget)                     # for a class you can't decorate

@cash.cache
def render(target):
    return f"rendered for {target.name}"
```

Only the class's code leaves the key. An instance is still keyed by its data,
so `render(RenderTarget("pdf"))` and `render(RenderTarget("svg"))` get separate
entries, and editing `RenderTarget` recomputes neither. `cash.opaque` returns the class
itself, so `isinstance` checks keep working. A subclass is not opaque unless you
mark it too, because it may have methods of its own that you edit. A
`functools.partial` can't be marked.

## Checking a marker

`cash.is_pure(func)` and `cash.is_stateful(func)` return whether a callable
carries the marker. They read the marker only; they don't analyse the code.

## Related

- [Side effects](../../decorator.md#side-effects): what cash reports, `# @cash:assume-safe`, `assume_safe=`, `strict=`.
- [Custom hashers](custom-hashers.md): change how an argument is keyed, rather than whether code is.
- [Purity markers reference](../../api/purity.md).
