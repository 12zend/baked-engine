#!/usr/bin/env bash
# Build the installable add-on zip into dist/. Usage: scripts/build_extension.sh [path/to/blender]
set -euo pipefail
BLENDER="${1:-${BLENDER:-blender}}"
cd "$(dirname "$0")/.."
mkdir -p dist
"$BLENDER" --factory-startup --command extension build --source-dir baked_engine --output-dir dist
