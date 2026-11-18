
import os
import sys
from pathlib import Path


# startar Cpp programmet

binary = Path(__file__).resolve().parent / "build/gd_classifier"
if not binary.is_file():
    raise SystemExit("bygg cpp först")
os.execv(str(binary), [str(binary), *sys.argv[1:]])
