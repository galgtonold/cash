# Controlling caching

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. How to change what cash does with one statement or
    one helper function.

Cash decides for each statement whether to cache it, and most of the time you
leave it alone. When you know something it cannot see, you have four levers:

- a `# @cash:` comment on one statement ([Annotations](../../annotations.md) is the
  full reference);
- `@stateful` on a helper function, so statements that call it always run;
- `%cash_on ttl=N`, a default expiry for every statement;
- `%cash_persist on`, which stores every statement on disk.

What cash already refuses or caches without being told is in
[What gets cached](../../notebook_caching_api.md#what-gets-cached). Check there
before adding a directive: you do not need `no-cache` on a file write, a POST or a
`datetime.now()`.

Start the notebook with the usual first cell:

```python { .nb-cell }
import cash
%cash_on
```

The examples below use these stand-ins for a slow price feed and a chat client:

```python { .nb-cell }
import time

def fetch_price(ticker):
    time.sleep(0.2)
    return 150.0

class ChatClient:
    def chat_postMessage(self, channel, text):
        return {"ok": True}

chat = ChatClient()
```

## Let a result expire: `ttl`

A price, a feed or anything else that changes on its own should not be served
forever. Give the statement a lifetime in seconds:

<!-- test:expect-badge rerun=EXECUTED -->
```python { .nb-cell }
# @cash:ttl=60
price = fetch_price("AAPL")    # served from the cache for a minute, then fetched again
```

`%cash_on ttl=3600` in the first cell sets a default lifetime for every
statement; a `# @cash:ttl=N` on a statement overrides it.

## Run a statement every time: `no-cache`

Use `no-cache` when the answer must be new on every run and cash cannot tell,
for example a read from a live source:

<!-- test:expect-badge rerun=EXECUTED -->
```python { .nb-cell }
# @cash:no-cache
live = fetch_price("AAPL")     # runs every time, nothing stored
```

The badge shows a plain `EXECUTED` row. It also gives a random draw a new value
each run (see [Randomness](../../known-limitations.md#randomness)).

## Keep a cheap result across restarts: `persist`

Only results that took more than 0.1 s to compute are stored on disk; cheaper
ones are kept in memory and computed again after a restart. When a cheap value
feeds something you want back instantly after a restart, store it anyway:

```python { .nb-cell }
# @cash:persist
tickers = sorted({"MSFT", "AAPL", "GOOG"})
```

`%cash_persist on` does this for every statement until `%cash_persist off`. It is
useful for a benchmark or a reproducible run, and wasteful for everyday work.
[Restarts and persistence](smart-persistence.md) explains what survives a restart.

## Cache a request that only reads: `assume-safe`

Cash runs every POST, PUT or upload every time, because a cache hit would skip
sending it. Some APIs use POST for plain queries, such as a search endpoint or an
LLM completion. Tell cash that skipping the request is harmless:

<!-- test:expect-badge rerun=EXECUTED -->
```python { .nb-cell }
class SearchSession:
    def post(self, url, json=None):
        time.sleep(0.2)
        return {"hits": [json["q"]]}

session = SearchSession()
```

<!-- test:expect-badge rerun=EXECUTED -->
```python { .nb-cell }
# @cash:assume-safe
# @cash:ttl=3600
hits = session.post("https://search.example.com", json={"q": "cash"})
```

It waives the side effect only; a statement that changes an object in place or
reads the clock still runs every time. Add a `ttl` when the answer can go stale.

## Stop caching calls inside a statement: `no-cache-calls`

Cash caches the expensive call inside a statement even when the statement itself
cannot be cached, such as `results.append(compute(x))`. If the function has an
effect cash cannot see, turn this off for the statement, or for a whole loop by
putting the comment on its header:

<!-- test:expect-badge rerun=EXECUTED -->
```python { .nb-cell }
results = []
# @cash:no-cache-calls
for t in ["AAPL", "MSFT"]:
    results.append(fetch_price(t))     # fetch_price runs on every run
```

## Mark a helper `@stateful` { #stateful-helpers }

A directive covers one statement. When a function's effect is the reason you call
it (it sends a message, writes to a database through a client library, updates a
dashboard), mark the function once, and every statement that calls it runs every
time:

```python { .nb-cell }
from cash import stateful

@stateful
def announce(text):
    return chat.chat_postMessage(channel="#runs", text=text)
```

<!-- test:expect-badge rerun=EXECUTED -->
```python { .nb-cell }
receipt = announce("model trained")    # badge: NOT CACHED - Calls @stateful function
```

<!-- claim: cash/purity.py:stateful @f86f4e92, cash/analysis/cacheability_decision.py:decide_cacheability @420335a6 -->
Without the marker, a slow `announce` call is cached and a re-run skips the
message: cash does not look inside `announce` for a chat client. Three things to
know:

- **Only direct calls count.** Cash checks the functions a statement calls by
  name, such as `announce(...)`. A method call such as `bot.announce(...)` is not
  checked, so marking a method does nothing. Call a module-level function
  instead, or put `# @cash:no-cache` on the statement.
- **It does not spread to callers.** A function that calls `announce` is not
  stateful itself, so a statement calling that wrapper is cached. Mark the
  wrapper too.
- **File writes need no marker.** A helper you wrote that writes a file
  (`fig.savefig(...)`, `df.to_csv(...)`, `open(p, "w")`), directly or through
  another of your functions, is detected: the statement runs every time and the
  badge says `Calls save(), which writes files`. Appending to a log file does not
  count. If such a write does not matter, mark the helper `@pure` and it caches
  again. Apart from this, `@pure` does not change what a notebook statement does.

## Randomness and estimators

`# @cash:allow-random` silences the warning about an unseeded draw without
changing what is cached; see [Randomness](../../known-limitations.md#randomness).
`# @cash:cache-fit` caches a bare `clf.fit(X, y)`; see
[Annotations](../../annotations.md#cashcache-fit).

## Related

- [Annotations](../../annotations.md): every directive, where to put it, and how
  several combine.
- [Reading the badge](../../badges.md): how each decision shows up.
- [Moving to a module](production-transition.md): the same controls on
  `@cash.cache` are decorator arguments.
