import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_TMP = ROOT / "result" / ".pytest_tmp"
TEST_TMP.mkdir(parents=True, exist_ok=True)
for var in ("TMP", "TEMP", "TMPDIR"):
    os.environ.setdefault(var, str(TEST_TMP))
tempfile.tempdir = str(TEST_TMP)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
