#!/usr/bin/env python3
"""Combine pure-image manifests while preserving init/regularization metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-prob", type=float, default=0.9999)
    parser.add_argument("--require-pred-target", action="store_true")
    args = parser.parse_args()

    frames = []
    for path_text in args.manifests:
        path = Path(path_text)
        frame = pd.read_csv(path)
        frame["source_manifest"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined[combined["final_prob"].astype(float) >= args.min_prob].copy()
    if args.require_pred_target:
        combined = combined[combined["final_pred"].astype(int) == combined["target_class"].astype(int)].copy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    metadata = {
        "input_manifests": args.manifests,
        "output": str(output),
        "min_prob": args.min_prob,
        "require_pred_target": args.require_pred_target,
        "rows_before_filter": int(before),
        "rows_after_filter": int(len(combined)),
        "by_init_regularization": {f"{k[0]}::{k[1]}": int(v) for k, v in combined.groupby(["init_mode", "regularization"]).size().to_dict().items()},
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
