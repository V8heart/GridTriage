#!/usr/bin/env bash
# GridTriage Physics Tier B environment (Python 3.11 + ANDES 2.0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_PHYSICS:-python3.11}"
VENV="${ROOT}/.venv-physics"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python 3.11 is required (set PYTHON_PHYSICS to its executable)." >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON}" -m venv "${VENV}"
fi

"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${ROOT}/requirements-physics.txt"

"${VENV}/bin/python" - <<'PY'
import sys
import andes

assert sys.version_info >= (3, 11)
assert andes.__version__ == "2.0.0"
print(f"Physics environment ready: Python {sys.version.split()[0]}, ANDES {andes.__version__}")
PY

echo "Activate with: source ${VENV}/bin/activate"
