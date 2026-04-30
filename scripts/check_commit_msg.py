#!/usr/bin/env python3
"""
Lightweight Conventional Commit checker for pre-commit commit-msg stage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)

PATTERN = re.compile(rf"^({'|'.join(ALLOWED_TYPES)})(\([a-z0-9._\-/]+\))?(!)?: [^\s].+")


def main() -> int:
    if len(sys.argv) < 2:
        print("Missing commit message file path.")
        return 2

    msg_file = Path(sys.argv[1])
    lines = msg_file.read_text(encoding="utf-8").splitlines()
    first_line = lines[0].strip() if lines else ""

    if PATTERN.match(first_line):
        return 0

    print("Invalid commit message format.")
    print("Expected: type(scope): subject or type!: subject")
    print(f"Allowed types: {', '.join(ALLOWED_TYPES)}")
    print(f"Got: {first_line!r}")
    print("Example: feat(mock): add batched generation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
