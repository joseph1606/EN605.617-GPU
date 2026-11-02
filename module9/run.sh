#!/usr/bin/env bash
# Runs every photo at the given noise level: ./run.sh <level 1-3> [seed]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output
for img in images/*.pgm; do
    ./assignment.exe "$img" "$@"
done
