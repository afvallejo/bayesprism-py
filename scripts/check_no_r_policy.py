from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _allowlisted_paths() -> set[str]:
    return {
        "notebooks/r_equivalence_deconvolution.ipynb",
    }


def _tracked_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    raw = proc.stdout.decode("utf-8", errors="ignore")
    return [Path(part) for part in raw.split("\0") if part]


def _forbidden_fragments() -> list[str]:
    return [
        "r" + "script",
        "sc" + "ran",
        "bioc" + "parallel",
        "r " + "package",
        "r " + "-> python",
    ]


def main() -> int:
    violations: list[str] = []
    fragments = _forbidden_fragments()
    forbidden_suffix = "." + "R"
    allowlisted = _allowlisted_paths()

    for rel_path in _tracked_files():
        if rel_path.as_posix() in allowlisted:
            continue

        if rel_path.suffix == forbidden_suffix:
            violations.append(f"{rel_path}: forbidden file suffix detected")
            continue

        abs_path = Path.cwd() / rel_path
        if not abs_path.is_file():
            continue

        content = abs_path.read_bytes()
        if b"\0" in content:
            continue

        text_lower = content.decode("utf-8", errors="ignore").lower()
        for fragment in fragments:
            if fragment in text_lower:
                violations.append(f"{rel_path}: contains forbidden token '{fragment}'")

    if violations:
        print("No-R policy violations found:")
        for item in sorted(set(violations)):
            print(f"- {item}")
        return 1

    print("No-R policy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
