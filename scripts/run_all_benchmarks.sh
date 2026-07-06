#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" src/evaluation/benchmark_text.py
"$PYTHON_BIN" src/evaluation/benchmark_image.py
"$PYTHON_BIN" src/evaluation/benchmark_audio.py
"$PYTHON_BIN" src/evaluation/plot_results.py
