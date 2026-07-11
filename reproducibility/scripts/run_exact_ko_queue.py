#!/usr/bin/env python3
"""Resumable queue for Q1 exact Step 1A K&O clean-start comparator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DEFAULT = "python"
MODELS = ["bbb_resnet50", "bbb_vgg19_bn", "bbb_densenet", "bbb_inception_v3"]
SPLIT_SEEDS = [1001, 1002, 1003]
CANDIDATE_SEEDS = [0, 1, 2, 3, 4]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["updated_at"] = now()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {text}\n")


def run_cmd(cmd: list[str], log_path: Path, state_path: Path, state: dict, env: dict[str, str]) -> None:
    append(log_path, "CMD " + " ".join(cmd))
    state["active_command"] = cmd
    write_json(state_path, state)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    state["active_pid"] = proc.pid
    write_json(state_path, state)
    rc = proc.wait()
    state["active_pid"] = None
    state["last_returncode"] = rc
    write_json(state_path, state)
    append(log_path, f"RC {rc}")
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=PYTHON_DEFAULT)
    p.add_argument("--output-root", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/exact_protocol"))
    p.add_argument("--log-dir", type=Path, default=Path("logs/exact_ko"))
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--split-seeds", default=",".join(str(x) for x in SPLIT_SEEDS))
    p.add_argument("--candidate-seeds", default=",".join(str(x) for x in CANDIDATE_SEEDS))
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--power-iters", type=int, default=12)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--alpha-grid", default="1,2,4,6,8")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--max-basis-images", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    split_seeds = [int(x.strip()) for x in args.split_seeds.split(",") if x.strip()]
    candidate_seeds = [int(x.strip()) for x in args.candidate_seeds.split(",") if x.strip()]
    out_root = args.output_root / "phase1a_ko_cleanstart_comparator"
    state_path = out_root / "ko_queue_state.json"
    log_path = args.log_dir / "ko_queue.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"

    state = {
        "status": "running",
        "queue": "q1_exact_ko_cleanstart_step1a",
        "output_root": str(out_root),
        "models": models,
        "split_seeds": split_seeds,
        "candidate_seeds": candidate_seeds,
        "k": args.k,
        "power_iters": args.power_iters,
        "tol": args.tol,
        "completed": [],
        "active_step": None,
        "active_pid": None,
    }
    write_json(state_path, state)
    try:
        for model in models:
            for split_seed in split_seeds:
                for candidate_seed in candidate_seeds:
                    step = f"ko_cleanstart:{model}:split{split_seed}:cand{candidate_seed}"
                    out = out_root / model / f"split_seed_{split_seed}" / f"candidate_seed_{candidate_seed}"
                    sentinel = out / "DONE"
                    if sentinel.exists() and not args.overwrite:
                        append(log_path, f"SKIP {step} existing {sentinel}")
                        state["completed"].append({"step": step, "artifact": str(out), "skipped": True})
                        write_json(state_path, state)
                        continue
                    state["active_step"] = step
                    write_json(state_path, state)
                    cmd = [
                        args.python,
                        "-u",
                        "experiments/hidden_jacobian_routing/run_exact_ko_cleanstart_comparator.py",
                        "--output-dir",
                        str(out),
                        "--model",
                        model,
                        "--split-seed",
                        str(split_seed),
                        "--candidate-seed",
                        str(candidate_seed),
                        "--k",
                        str(args.k),
                        "--max-k",
                        str(args.k),
                        "--power-iters",
                        str(args.power_iters),
                        "--tol",
                        str(args.tol),
                        "--alpha-grid",
                        args.alpha_grid,
                    ]
                    if args.max_images >= 0:
                        cmd.extend(["--max-images", str(args.max_images)])
                    if args.max_basis_images >= 0:
                        cmd.extend(["--max-basis-images", str(args.max_basis_images)])
                    if args.overwrite:
                        cmd.append("--overwrite")
                        cmd.append("--overwrite-basis")
                    run_cmd(cmd, log_path, state_path, state, env)
                    state["completed"].append({"step": step, "artifact": str(out), "skipped": False})
                    write_json(state_path, state)
        state["status"] = "complete_step1a_ko_cleanstart"
        state["active_step"] = None
        write_json(state_path, state)
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        write_json(state_path, state)
        append(log_path, "FAILED " + str(exc))
        raise


if __name__ == "__main__":
    main()
