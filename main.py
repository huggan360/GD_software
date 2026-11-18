"""smart_fridge_edge invokes Python; inference itself runs in C++."""
import os
import sys
from pathlib import Path

binary = Path(__file__).resolve().parent / "build/gd_classifier"
if not binary.is_file():
    raise SystemExit("Build the C++ program first: bash compile/build.sh")
os.execv(str(binary), [str(binary), *sys.argv[1:]])
