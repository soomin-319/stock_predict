import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(ROOT / "result" / ".pytest_tmp" / f"run-{os.getpid()}")


def pytest_sessionfinish(session, exitstatus):
    if exitstatus != 0 or os.getenv("KEEP_TEST_ARTIFACTS") == "1":
        return
    factory = getattr(session.config, "_tmp_path_factory", None)
    if factory is None:
        return
    session_root = Path(factory.getbasetemp()).resolve()
    shared_root = (ROOT / "result" / ".pytest_tmp").resolve()
    if session_root != shared_root and shared_root in session_root.parents:
        shutil.rmtree(session_root, ignore_errors=True)
