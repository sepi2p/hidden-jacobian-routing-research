#!/usr/bin/env python3
"""Regenerate table-ready LaTeX tabular files from frozen CSV inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def escape_cell(cell: str) -> str:
    # Frozen cells intentionally retain the mathematical LaTeX used by the paper.
    return cell.strip()


def build_one(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"empty table input: {source}")
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    alignment = "l" + "r" * (width - 1)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(escape_cell(x) for x in normalized[0]) + r" \\")
    lines.append("\\midrule")
    for row in normalized[1:]:
        lines.append(" & ".join(escape_cell(x) for x in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/table_inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated/tables")
    args = parser.parse_args()
    sources = sorted(args.input_dir.glob("table_*.csv"))
    if not sources:
        raise SystemExit(f"no table inputs found under {args.input_dir}")
    for source in sources:
        build_one(source, args.output_dir / f"{source.stem}.tex")
    print(f"generated {len(sources)} tables in {args.output_dir}")


if __name__ == "__main__":
    main()
