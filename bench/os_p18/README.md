# p18 first run: Windows vs Linux (WSL) on one machine

Throwaway bench branch. Cash itself is the claude/batch19 code at b9fc5f78b.
`run_p18.py --venv VENV --out OUT` replays p18-clinical (from correctness-set snapshot set-194b174)
5 alternating plain/cash rounds, plus 2 rounds with glibc's malloc thresholds pinned on Linux.
Results: OUT/SUMMARY.md; progress in OUT/progress.txt; a restart skips finished runs.

Env (same on both): Python 3.13, numpy==2.5.4 pandas==3.1.1 scipy==1.18.1 seaborn==0.13.2 matplotlib==3.11.2
pyarrow ipykernel==7.4.0 jupyter-client==8.10.0 psutil, cash installed editable from this checkout with the
C extension built (`python hatch_build.py`; `cash._native.status_line()` says "on").
