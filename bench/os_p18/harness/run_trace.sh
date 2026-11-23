#!/usr/bin/env bash
# usage: run_trace.sh PROJECT_DIR TRACE RUNS_DIR [VENV]
# Correctness-set run of one trace item: plain1, cash1 (verify ON), plain2, cash2 (verify off),
# alternated, one at a time, into a local temp dir that is copied to RUNS_DIR after each run.
# Then compare.py (correctness over all four) and speed.py (plain vs cash; cash2 is the clean timing).
# Harness interpreter: $CS_PY, else /home/user/r35/venv-harness/bin/python.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${1:?usage: run_trace.sh PROJECT_DIR TRACE RUNS_DIR [VENV]}" && pwd)" || { echo "run_trace: no project dir $1" >&2; exit 2; }
[ -n "$PROJ" ] || { echo "run_trace: no project dir $1" >&2; exit 2; }
TRACE="${2:?trace}"
RUNS="${3:?runs dir}"
[ -n "${4:-}" ] && export R34_VENV="$4"
PY="${CS_PY:-/home/user/r35/venv-harness/bin/python}"
VERIFY="CASH_HITPATH_VERIFY=1 CASH_CWATCH_VERIFY=1"
mkdir -p "$RUNS"
LOCAL="$(mktemp -d /tmp/cs-runs-XXXX)"
status=0
for run in plain1 cash1 plain2 cash2; do
  mode="${run%[12]}"
  if [ "$run" = cash1 ]; then cenv="$VERIFY"; else cenv=""; fi
  echo "== $run ($mode${cenv:+, verify on}) $(date -u +%H:%M:%S)"
  if ! CS_CASH_ENV="$cenv" "$PY" "$H/replay.py" "$TRACE" "$PROJ" "$mode" "$LOCAL/$run.jsonl" > "$LOCAL/$run.log" 2>&1; then
    echo "   replay $run exited non-zero; see $RUNS/$run.log"; status=1
  fi
  cp "$LOCAL/$run.jsonl" "$LOCAL/$run.log" "$RUNS/" 2>/dev/null
  tail -n 2 "$RUNS/$run.log" | sed 's/^/   /'
done
rm -rf "$LOCAL"
"$PY" "$H/compare.py" "$RUNS" || status=1
"$PY" "$H/speed.py" "$RUNS" || status=1
"$PY" "$H/stats.py" "$RUNS" || status=1
echo "== done $(date -u +%H:%M:%S): $RUNS/compare.md $RUNS/speed.md $RUNS/stats.md"
exit $status
