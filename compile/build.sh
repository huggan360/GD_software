#!/usr/bin/env bash
set -euo pipefail

# Kräver CMake, en C++-kompilator, OpenCV, LibTorch och HailoRT.
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -z "${TORCH_ROOT:-}" ]]; then TORCH_ROOT=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)'); fi
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" "$@"
cmake --build "$root/build" --parallel 2
