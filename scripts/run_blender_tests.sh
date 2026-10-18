#!/usr/bin/env bash
# Run the tests that need Blender. Usage: scripts/run_blender_tests.sh [path/to/blender]
set -euo pipefail
BLENDER="${1:-${BLENDER:-blender}}"
cd "$(dirname "$0")/.."
status=0
for t in tests/blender/test_*.py; do
  echo "== $t"
  "$BLENDER" -b --factory-startup --python "$t" 2>&1 | grep -E "^(PASS|FAIL|RESULT)|Traceback|Error" || true
  "$BLENDER" -b --factory-startup --python "$t" >/dev/null 2>&1 || status=1
done
exit $status
