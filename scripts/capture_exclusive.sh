#!/usr/bin/env bash
# Run a queue of dataset.run_capture argument lines one at a time on otherwise idle GPUs.
# Before each session both GPUs must be idle (no compute process, power below
# IDLE_MAX_W) for IDLE_HOLD_S straight; after each session the telemetry is
# checked for foreign GPU processes and flagged in the log.
#
#   bash scripts/capture_exclusive.sh scripts/queues/ltma_v2.txt --dry-run
#   nohup bash scripts/capture_exclusive.sh scripts/queues/ltma_v2.txt --confirm-shared-gpu-safe \
#       > dataset/real/logs/capture_ltma_v2.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
IDLE_MAX_W="${IDLE_MAX_W:-40}"
IDLE_HOLD_S="${IDLE_HOLD_S:-120}"
POLL_S=10
DRY_RUN=0
CONFIRMED=0
QUEUE="${1:?usage: capture_exclusive.sh <queue-file> [--dry-run|--confirm-shared-gpu-safe]}"
shift

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --confirm-shared-gpu-safe) CONFIRMED=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [[ "$DRY_RUN" != "1" && "$CONFIRMED" != "1" ]]; then
  echo "--confirm-shared-gpu-safe is required" >&2
  exit 2
fi

cd "$ROOT"
mapfile -t configs < <(grep -v '^\s*#' "$QUEUE" | grep -v '^\s*$')
common="--confirm-shared-gpu-safe"

gpus_idle() {
  local apps power
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)"
  [[ "$apps" == "0" ]] || return 1
  while read -r power; do
    awk -v p="$power" -v m="$IDLE_MAX_W" 'BEGIN { exit !(p <= m) }' || return 1
  done < <(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
}

wait_for_idle() {
  local held=0
  while (( held < IDLE_HOLD_S )); do
    if gpus_idle; then
      held=$(( held + POLL_S ))
    else
      if (( held > 0 )); then echo "[$(date -Is)] GPU busy again, idle timer reset"; fi
      held=0
    fi
    sleep "$POLL_S"
  done
  echo "[$(date -Is)] both GPUs idle for ${IDLE_HOLD_S}s"
}

latest_session() {
  ls -td dataset/real/sessions/s-* 2>/dev/null | head -1
}

check_contamination() {
  local session="$1"
  "$PYTHON_BIN" - "$session" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd

session = Path(sys.argv[1])
df = pd.read_csv(session / "telemetry.csv", usecols=["gpu_id", "observed_n_procs", "gt_label"])
notes = []
for gid, part in df.groupby("gpu_id"):
    n = pd.to_numeric(part["observed_n_procs"], errors="coerce").fillna(0)
    if part["gt_label"].iloc[0] == "normal_idle" and (n > 0).any():
        notes.append(f"gpu{gid}: idle companion had a GPU process in {int((n > 0).sum())} rows")
    elif (n > 1).any():
        notes.append(f"gpu{gid}: more than one GPU process in {int((n > 1).sum())} rows")
print(json.dumps({"session": session.name, "clean": not notes, "notes": notes}))
PY
}

for cfg in "${configs[@]}"; do
  cmd="$PYTHON_BIN -m dataset.run_capture $common $cfg"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s --dry-run\n' "$cmd"
    continue
  fi
  wait_for_idle
  echo "[$(date -Is)] start: $cfg"
  read -r -a argv <<<"$cmd"
  if "${argv[@]}"; then
    echo "[$(date -Is)] done: $cfg"
    check_contamination "$(latest_session)"
  else
    echo "[$(date -Is)] FAILED: $cfg"
  fi
done
echo "[$(date -Is)] all captures finished: $QUEUE"
