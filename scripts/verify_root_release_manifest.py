#!/usr/bin/env python3
"""Check all tracked publication files against the repository-wide manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

try:
    from scripts.build_root_release_manifest import ROOT, MANIFEST, public_files
except ModuleNotFoundError:  # direct invocation from the scripts directory
    from build_root_release_manifest import ROOT, MANIFEST, public_files


def verify(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    with (root / MANIFEST.name).open(newline="", encoding="utf-8") as handle:
        recorded = list(csv.DictReader(handle))
    expected_paths = {path.relative_to(root).as_posix() for path in public_files(root)}
    recorded_paths = [row["path"] for row in recorded]
    if len(recorded_paths) != len(set(recorded_paths)):
        errors.append("duplicate manifest path")
    if expected_paths != set(recorded_paths):
        errors.append(f"manifest path coverage differs: unlisted={sorted(expected_paths - set(recorded_paths))}, missing={sorted(set(recorded_paths) - expected_paths)}")
    for row in recorded:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"unsafe manifest path: {row['path']}")
            continue
        path = root / rel
        if not path.is_file():
            errors.append(f"missing file: {row['path']}")
            continue
        payload = path.read_bytes()
        if len(payload) != int(row["bytes"]) or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            errors.append(f"checksum/size mismatch: {row['path']}")
    return errors


if __name__ == "__main__":
    failures = verify()
    for issue in failures:
        print("[FAIL]", issue)
    if failures:
        raise SystemExit(1)
    print("[PASS] repository file manifest; not an independent experimental rerun")
