import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(ROOT / "result" / ".pytest_tmp" / f"run-{os.getpid()}")
