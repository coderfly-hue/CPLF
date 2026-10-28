"""Regenerate the SHA-256 list for the manuscript-linked protocol directory."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CPLF LLM protocols io complete package" / "CPLF LLM protocols io FORMAL RELEASE"
MANIFEST = SOURCE / "MANIFEST_SHA256.csv"


def build() -> int:
    names = sorted(str(path.relative_to(SOURCE)) for path in SOURCE.rglob("*")
                   if path.is_file() and path != MANIFEST and
                   "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})
    rows = []
    for rel in names:
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path: {path}")
        payload = (SOURCE / path).read_bytes()
        rows.append((rel, len(payload), hashlib.sha256(payload).hexdigest()))
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "bytes", "sha256"))
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print(f"Published manifest updated: {build()} files")
