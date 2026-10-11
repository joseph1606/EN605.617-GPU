#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PATH:/usr/local/cuda/bin"
nvcc -O2 -o assignment.exe assignment.cu
