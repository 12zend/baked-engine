#!/usr/bin/env bash
# Run the tests that need Blender. Usage: scripts/run_blender_tests.sh [path/to/blender]
set -uo pipefail
BLENDER="${1:-${BLENDER:-blender}}"
cd "$(dirname "$0")/.."
status=0
for t in tests/blender/test_*.py; do
  echo "== $t"
  out="$("$BLENDER" -b --factory-startup --python "$t" 2>&1)" || status=1
  printf '%s\n' "$out" | grep -E "^(PASS|FAIL|RESULT)|Traceback|Error" || true
done
exit $status
