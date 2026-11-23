#!/usr/bin/env bash
# usage: run_all.sh PROJECT_DIR [TRACE]
# Replays PROJECT_DIR/trace.json plain, cash, plain, cash (one after another)
# into PROJECT_DIR/runs/, then runs compare.py and speed.py on that folder.
# Python for the harness itself: $R34_PY, else /home/user/r34/venv-harness/bin/python.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${1:?usage: run_all.sh PROJECT_DIR [TRACE]}" && pwd)"
TRACE="${2:-$PROJ/trace.json}"
PY="${R34_PY:-/home/user/r34/venv-harness/bin/python}"
RUNS="${R35_RUNS:-$PROJ/runs}"  # R35_RUNS overrides (cold runs)
mkdir -p "$RUNS"
# round 35: replays append to a local folder (appends on the shared mount lost records in round 34),
# copied to $RUNS after each run.
LOCAL="$(mktemp -d /tmp/r35-runs-XXXX)"
status=0
for run in plain1 cash1 plain2 cash2; do
  mode="${run%[12]}"
  echo "== $run ($mode) $(date -u +%H:%M:%S)"
  if ! "$PY" "$H/replay.py" "$TRACE" "$PROJ" "$mode" "$LOCAL/$run.jsonl" > "$LOCAL/$run.log" 2>&1; then
    echo "   replay $run exited non-zero; see $RUNS/$run.log"; status=1
  fi
  cp "$LOCAL/$run.jsonl" "$LOCAL/$run.log" "$RUNS/" 2>/dev/null
  tail -n 3 "$RUNS/$run.log" | sed 's/^/   /'
done
rm -rf "$LOCAL"
"$PY" "$H/compare.py" "$RUNS" || status=1
"$PY" "$H/speed.py" "$RUNS" || status=1
echo "== done $(date -u +%H:%M:%S): $RUNS/compare.md $RUNS/speed.md"
exit $status
