#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/stage5.yaml}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/run_protocol.py" \
  --config "${CONFIG_PATH}" \
  --protocol protocol_2 \
  --include-sensitivity

# Paper Sec. 5.2 protocol two: mixed training, known/unseen tests, and sensitivity.
