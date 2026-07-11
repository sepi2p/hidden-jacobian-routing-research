#!/usr/bin/env python3
"""Refresh checksums for frozen lightweight artifacts and registries."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = sorted((ROOT / "artifacts").rglob("*")) + sorted((ROOT / "reproducibility/configs").rglob("*"))
    files = [path for path in paths if path.is_file()]
    lines = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in files]
    (ROOT / "reproducibility/SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} checksums")


if __name__ == "__main__":
    main()
