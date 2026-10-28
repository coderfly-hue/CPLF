#!/usr/bin/env python3
"""Build the repository-wide SHA-256 manifest from publishable source files."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST_SHA256.csv"
EXCLUDED_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist", "outputs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def public_files(root: Path = ROOT) -> list[Path]:
    return sorted((path for path in root.rglob("*")
                   if path.is_file() and path != root / MANIFEST.name
                   and not any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts)
                   and path.suffix not in EXCLUDED_SUFFIXES),
                  key=lambda p: p.relative_to(root).as_posix())


def build(root: Path = ROOT) -> int:
    manifest = root / MANIFEST.name
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "bytes", "sha256"))
        for path in public_files(root):
            data = path.read_bytes()
            writer.writerow((path.relative_to(root).as_posix(), len(data), hashlib.sha256(data).hexdigest()))
    return len(public_files(root))


if __name__ == "__main__":
    print(f"Repository SHA-256 manifest updated: {build()} files")
