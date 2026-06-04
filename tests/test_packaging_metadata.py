import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def _requirement_lines_from_pyproject() -> list[str]:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return list(data["project"]["dependencies"])


def _requirement_lines_from_requirements_txt() -> list[str]:
    lines: list[str] = []
    for raw_line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _applicable_numpy_specifiers(lines: list[str], python_version: str) -> list[str]:
    env = {"python_version": python_version}
    specs: list[str] = []
    for line in lines:
        req = Requirement(line)
        if req.name.lower() != "numpy":
            continue
        if req.marker is None or req.marker.evaluate(env):
            specs.append(str(req.specifier))
    return specs


def _specifier_parts(specs: list[str]) -> set[str]:
    assert len(specs) == 1
    return set(specs[0].split(","))


def test_numpy_requirement_selects_python314_wheel_compatible_release():
    for lines in (_requirement_lines_from_requirements_txt(), _requirement_lines_from_pyproject()):
        specs = _applicable_numpy_specifiers(lines, "3.14")

        assert _specifier_parts(specs) == {">=2.3", "<2.4"}


def test_numpy_requirement_preserves_python312_release_range():
    for lines in (_requirement_lines_from_requirements_txt(), _requirement_lines_from_pyproject()):
        specs = _applicable_numpy_specifiers(lines, "3.12")

        assert _specifier_parts(specs) == {">=2.0", "<2.3"}
