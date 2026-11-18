#!/usr/bin/env bash
set -euo pipefail
# Needs CMake, a C++ compiler, OpenCV development libraries and CPU LibTorch.
# The Pi also needs its installed HailoRT. For a PC, add -DGD_HAILO=OFF.
# Export models with the same PyTorch version used by the runtime libraries.
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Point TORCH_ROOT at LibTorch, or use the torch package in your active environment.
if [[ -z "${TORCH_ROOT:-}" ]]; then TORCH_ROOT=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)'); fi
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" "$@"
cmake --build "$root/build" --parallel 2
