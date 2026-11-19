#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
package="$root/compile/build/openblas"
mkdir -p "$package" "$root/build/lib"
cd "$package"

# Hämta Debian-paketet och packa upp det utan att ändra systemets bibliotek.
apt-get download libopenblas0-pthread
dpkg-deb -x libopenblas0-pthread_*.deb unpacked
library=$(find unpacked -name libblas.so.3 -print -quit)
cp -a "$(dirname -- "$library")/." "$root/build/lib/"

# Bygg programmet med OpenBLAS. Biblioteken följer med i build/lib.
cmake -S "$root" -B "$root/build"
cmake --build "$root/build" --parallel 2
