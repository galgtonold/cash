# `zzprobe_*` / `zzmeas_*` — standalone scripts, not tests

These are **not collected by pytest**. `python_files` matches `test_*.py`, and
nothing here starts with `test_`, so they never run in a suite. Run one
directly:

```bash
python tests/test_notebook_integration/zzmeas_highn.py
```

## Why they are in git at all

Most probes are scratch and get deleted once the behaviour they investigated is
pinned by a real guard test. A probe is kept **only** when something durable
cites it:

| File | Cited by | Why it has to survive |
|---|---|---|
| `zzmeas_cas261_band.py`, `zzmeas_cas261_floor.py`, `zzmeas_cas261_steady.py`, `zzmeas_cas261_storecost.py` | `src/cash/notebook/call_unit.py` (the cost-floor constants) | The constants in that module are *fitted*, not chosen. These scripts are how the numbers were obtained, so they are what you re-run before changing one. |
| `zzmeas_highn.py` | CAS-264 | Produces the measurement table the ticket's severity rests on — including the finding that the overhead is constant (~25–50 ms) rather than proportional, which is what downgraded it. |
| `zzprobe_hidden_state_arg.py` | `test_call_unit_acceptance.py` | Exercises the hidden-state-behind-a-bare-`Name` case that motivated the dynamic occurrence counter. |
| `zzprobe_impure_callee_statement.py` | CAS-243, CAS-246 | The measurement showing the STATEMENT path *already* freezes an impure callee — the load-bearing evidence that interception adds no new class of defect, and CAS-246's whole premise. |

## The rule

If you write a probe and it settles a question, either **promote it to a real
`test_*.py` guard** or **delete it**. Leave it here only if a ticket or a source
comment now depends on it — and then say so in the table above, or the next
cleanup will delete it as scratch.

A probe with no citation is not evidence anyone can find. That is the state this
directory was in before 2026-08-07: 32 untracked scripts, of which 25 were
superseded by guards that already existed.
