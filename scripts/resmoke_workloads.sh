#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DRY_RUN=0
CONFIRMED=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --confirm-shared-gpu-safe) CONFIRMED=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$DRY_RUN" != "1" && "$CONFIRMED" != "1" ]]; then
  echo "--confirm-shared-gpu-safe is required for resmoke" >&2
  exit 2
fi

cd "$ROOT"
commands=(
  "$PYTHON_BIN -m dataset.run_capture --workload llm_pretrain_ddp --preset small --seq-len 256 --duration 120 --warmup-s 0 --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload llm_pretrain_fsdp --preset small --seq-len 256 --duration 120 --warmup-s 0 --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload llm_finetune --preset small --seq-len 256 --duration 120 --warmup-s 0 --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload swma_basic --duration 120 --warmup-s 0 --period 1 --matrix-size 4096 --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload swma_piggyback --duration 120 --warmup-s 0 --host llm --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload swma_mimicry --duration 120 --warmup-s 0 --host llm --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload ltma --duration 120 --warmup-s 0 --host llm --aux-width 2048 --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload cryptojacking --duration 120 --warmup-s 0 --matrix-size 4096 --confirm-shared-gpu-safe"
)

for command in "${commands[@]}"; do
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$command"
  else
    read -r -a argv <<<"$command"
    "${argv[@]}"
  fi
done
