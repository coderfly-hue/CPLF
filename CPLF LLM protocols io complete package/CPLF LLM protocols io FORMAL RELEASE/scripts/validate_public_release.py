#!/usr/bin/env python3
"""Compatibility entry point for the stronger repository release audit.

Presence of filenames alone is insufficient; this now checks hashes, public
IDs, score arithmetic and seed-level consistency as well.
"""
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "scripts"))
from verify_formal_release import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
