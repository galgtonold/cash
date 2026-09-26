# Use case: data science

!!! info "Applies to: notebook"
    A worked notebook: load, clean, build features, train and evaluate, then
    change one line and see what runs again.

A data science notebook runs the same cells over and over while you change one
thing at a time: a filter, a feature, a hyperparameter. Without caching, every
change pays for the whole chain again, and so does every kernel restart. With
cash, only the statements whose code or inputs changed run.

## The notebook

### Cell 1: setup

```python { .nb-cell }
import cash
%cash_on
```

### Cell 2: load the data

```python { .nb-cell }
import pandas as pd

customers = pd.read_csv("customers.csv")
transactions = pd.read_csv("transactions.csv")

print(f"Customers: {len(customers)}, Transactions: {len(transactions)}")
```

cash tracks both files. Change `customers.csv` and the statements that read it
run again. Open the badge and the row names the file:

<iframe class="cash-badge" title="cash badge example: customers.csv changed" src="/_badges/miss_file_changed_customers.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<p class="cash-badge-caption">Click a badge to open it.</p>

### Cell 3: clean

<!-- test:expect-badge rerun=EXECUTED -->
```python { .nb-cell }
customers = customers.dropna(subset=["email", "signup_date"])
customers = customers.assign(
    signup_date=pd.to_datetime(customers["signup_date"])
)
transactions = transactions.assign(
    date=pd.to_datetime(transactions["date"])
)
```

Each statement is cached on its own. Change the `dropna` and only it, and what
depends on it, runs again.

<!-- claim: cash/analysis/mutations.py:MutationVisitor @7c19f318 broad="the subscript-store verdict is one branch of the visitor, read together with the outputs rule" -->
!!! tip "Use `.assign()` for columns worth caching"
    `df["col"] = ...` on a frame from an earlier cell changes that frame in
    place, so it runs every time. `df = df.assign(col=...)` makes a new frame
    and is cached. On a frame built in the same cell, both are cached. It only
    matters for expensive columns.

### Cell 4: features

```python { .nb-cell }
tx_features = transactions.groupby("customer_id").agg(
    total_spend=("amount", "sum"),
    avg_spend=("amount", "mean"),
    tx_count=("amount", "count"),
    last_purchase=("date", "max"),
).reset_index()

df = customers.merge(tx_features, on="customer_id", how="left")
```

### Cell 5: train

```python { .nb-cell }
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = ["total_spend", "avg_spend", "tx_count"]
X = df[features].fillna(0)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42
).fit(X_train, y_train)
```

The fitted model is cached like any other value, keyed on the data and the
hyperparameters, and a slow fit is on disk after a kernel restart. Keep the fit
in one assignment: a bare `model.fit(X_train, y_train)` changes `model` in place
and runs every time (see [`# @cash:cache-fit`](../../annotations.md#cashcache-fit)).
Pass `random_state`: an unseeded fit is cached too, but as a frozen replay, and
cash warns about it.

### Cell 6: evaluate

```python { .nb-cell }
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
```

## Change one line

Switch the merge in cell 4 from `how="left"` to `how="inner"` and run the
notebook again:

- Cells 2 and 3 are `CACHED`.
- In cell 4 the `groupby` is `CACHED`; only the merge runs.
- Cells 5 and 6 run, because `df` changed.

<iframe class="cash-badge" title="cash badge example: the notebook after one changed line" src="/_badges/workflows_mixed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

The loads and the aggregation, the slow part, cost nothing on this run.

## Habits that keep it fast

1. **Give slow loads their own cell**, so changes further down never touch them.
2. **Build new frames instead of changing old ones**: `df = df.sort_values(...)`,
   not `df.sort_values(..., inplace=True)`, when the frame came from an earlier
   cell.
3. **Seed your randomness**: `random_state=42`, `np.random.seed(42)`,
   `df.sample(100, random_state=42)`. cash does not see every random draw, and an
   unseeded one is replayed, not redrawn. See
   [Randomness](../../known-limitations.md#randomness).
4. **Pass the date in.** `datetime.now()` in a statement runs every time, but
   inside a helper function it is not seen, and the call is cached with the
   first value. Compute the timestamp in its own statement and pass it in.

## Related

- [Notebook guide](../../notebook_caching_api.md): what is cached and where the
  cache lives.
- [Reading the badge](../../badges.md): what each row means.
- [Restarts and persistence](../feature-guides/smart-persistence.md): what
  survives a kernel restart.
- [Moving to a module](../feature-guides/production-transition.md): when the
  notebook becomes a script.
