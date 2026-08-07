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

### Round-14 gate adjudication (2026-08-07)

Five user-testing agents reported 8 WRONG and 1 BLOCKING. One survived
adjudication (CAS-270). These probes are why the other eight did not, and they
are cited from CAS-140's round log — a "not reproduced" verdict is only worth
anything if the attempt is inspectable.

| File | Finding | Verdict |
|---|---|---|
| `zzprobe_r14p1_uncacheable_reads_stale.py` | an uncacheable plot cell reads a stale upstream | not reproduced (minimal shape) |
| `zzprobe_r14p1_faithful_chain.py` | the same, with in-place column assignment and a >8 MiB frame | not reproduced |
| `zzprobe_r14p3_filewrite_reorder.py` | file-write side effect dropped on loop reorder, one array shape | discriminator refuted — all four arms behave identically, including the reporter's own control |
| `zzprobe_r14p4_accumulator.py` | log doubled on re-run; accumulator empty after restart | neither reproduced; both inverted |
| `zzprobe_r14p4_keyerror.py` | a mutating helper's `KeyError` swallowed | not reproduced — the body never re-executed, so there was no error to swallow |
| `zzprobe_r14p5_module_edit.py` | editing a helper module serves stale data | not reproduced in 4 arms (module-attribute vs bare name, at two body costs) |

Read these before re-investigating any of those findings: each already rules
out the obvious mechanism, and two of them record a wrong hypothesis of my own
alongside the right answer.

## The rule

If you write a probe and it settles a question, either **promote it to a real
`test_*.py` guard** or **delete it**. Leave it here only if a ticket or a source
comment now depends on it — and then say so in the table above, or the next
cleanup will delete it as scratch.

A probe with no citation is not evidence anyone can find. That is the state this
directory was in before 2026-08-07: 32 untracked scripts, of which 25 were
superseded by guards that already existed.
