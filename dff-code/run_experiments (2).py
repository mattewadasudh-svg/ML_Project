"""
run_experiments.py -- run named experiments (one per GPU, in parallel), evaluate each, summarise.

  python run_experiments.py dff_full dff_noaux        # train both, then evaluate both
  python run_experiments.py --summary                 # table of every finished run in runs/
  python run_experiments.py dff_full --dry_run        # just print the commands

Every experiment writes to runs/<name>/ (log.jsonl, last.pt, best.pt, done.json, results.json).
A run whose done.json exists is skipped for training, so re-running this script is safe.
best.pt = the epoch with the highest FF++ *validation* video AUC; it is evaluated on FF++ test and
on the unseen datasets CDF-v1 / CDF-v2 (test lists, all 32 frames per video).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FF_ROOT = os.environ.get("FF_ROOT", "/kaggle/input/datasets/saisdsaiiitbh/ff-crops")
EV_ROOT = os.environ.get("EV_ROOT", "/kaggle/input/datasets/saisdsaiiitbh/eval-corps")

# settings shared by ALL Small experiments, so that comparisons between them are fair
COMMON = [
    "--crops_root", FF_ROOT,
    "--frames_per_video", "16",
    "--val_frames", "8",
    "--batch_size", "24",
    "--num_workers", "4",
    "--ckpt_every", "500",
    "--log_every", "50",
    "--max_hours", "8",
]

EXPERIMENTS = {
    "dff_full": [
        "--backbone", "small",
        "--epochs", "12",
    ],
    "dff_noaux": [
        "--backbone", "small",
        "--epochs", "12",
        "--no_aux",
    ],
    "dff_ls01": [
        "--backbone", "small",
        "--epochs", "12",
        "--label_smoothing", "0.1",
    ],
    "dff_full_s1": [
        "--backbone", "small",
        "--epochs", "12",
        "--seed", "1",
    ],
    "dff_ls01_s1": [
        "--backbone", "small",
        "--epochs", "12",
        "--label_smoothing", "0.1",
        "--seed", "1",
    ],
    "lora": [
        "--backbone", "small",
        "--epochs", "12",
        "--variant", "lora",
    ],
    "linear": [
        "--backbone", "small",
        "--epochs", "12",
        "--variant", "linear",
    ],
    "dff_base": [
        "--backbone", "base",
        "--epochs", "12",
    ],
}

EVAL_SETS = [f"{FF_ROOT}/ffpp_manifest_split.csv:test",
             f"{EV_ROOT}/cdf1_manifest.csv:all",
             f"{EV_ROOT}/cdf2_manifest.csv:all"]


def train_cmd(name):
    return [sys.executable, "train.py", *COMMON, "--out_dir", f"runs/{name}", *EXPERIMENTS[name]]


def opt(name, flag, default):
    extra = EXPERIMENTS[name]
    return extra[extra.index(flag) + 1] if flag in extra else default


def eval_cmd(name):
    return [sys.executable, "evaluate_dff.py", "--ckpt", f"runs/{name}/best.pt",
            "--backbone", opt(name, "--backbone", "small"), "--variant", opt(name, "--variant", "dff"),
            "--sets", *EVAL_SETS, "--num_workers", "4", "--out", f"runs/{name}/results.json"]


def launch(name, gpu, dry):
    cmd = train_cmd(name)
    print(f"[GPU {gpu}] {name}: {' '.join(cmd)}", flush=True)
    if dry:
        return None
    Path("runs").mkdir(exist_ok=True)
    log = open(f"runs/{name}.log", "a")
    return subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1"),
                            stdout=log, stderr=subprocess.STDOUT)


def evaluate(name, gpu, dry):
    if dry:
        print(f"[GPU {gpu}] eval {name}: {' '.join(eval_cmd(name))}")
        return
    if not Path(f"runs/{name}/done.json").exists():
        print(f"{name}: training INCOMPLETE (no done.json) -- re-run this script to resume; not evaluating")
        return
    with open(f"runs/{name}/eval.log", "w") as log:
        subprocess.run(eval_cmd(name), env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                       stdout=log, stderr=subprocess.STDOUT)
    print(f"{name}: evaluation written to runs/{name}/results.json", flush=True)


def last_line(path):
    try:
        lines = [l for l in open(path).read().replace("\r", "\n").splitlines() if l.strip()]
        return lines[-1][:200] if lines else "(no output yet)"
    except OSError:
        return "(no log yet)"


def summary():
    rows = []
    for p in sorted(Path("runs").glob("*/results.json")):
        r = json.load(open(p))
        g = lambda key, m: 100 * r[key][m] if key in r else float("nan")
        keys = list(r)                          # order: ffpp test, cdf1, cdf2
        rows.append((p.parent.name, *[100 * r[k]["video_auc"] for k in keys],
                     *[100 * r[k]["video_eer"] for k in keys]))
    print(f"{'run':14s} {'FF++ AUC':>9s} {'CDF1 AUC':>9s} {'CDF2 AUC':>9s} | {'FF++ EER':>9s} {'CDF1 EER':>9s} {'CDF2 EER':>9s}")
    for row in rows:
        print(f"{row[0]:14s} " + " ".join(f"{v:9.2f}" for v in row[1:4]) + " | " + " ".join(f"{v:9.2f}" for v in row[4:7]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--gpus", type=int, default=0, help="default: all visible GPUs")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        return summary()
    bad = [n for n in args.names if n not in EXPERIMENTS]
    if bad:
        sys.exit(f"unknown experiments {bad}; known: {list(EXPERIMENTS)}")
    n_gpus = args.gpus
    if not n_gpus:
        import torch
        n_gpus = max(1, torch.cuda.device_count())
    pending = [n for n in args.names if not Path(f"runs/{n}/done.json").exists() or args.dry_run]
    for n in args.names:
        if n not in pending:
            print(f"{n}: already trained (done.json) -- skipping training")
    running, last_report = {}, time.time()
    while pending or running:
        for gpu in range(n_gpus):
            if gpu not in running and pending:
                name = pending.pop(0)
                running[gpu] = (name, launch(name, gpu, args.dry_run))
        if args.dry_run:
            for gpu, (name, _) in running.items():
                evaluate(name, gpu, True)
            break
        time.sleep(20)
        if time.time() - last_report > 120:          # progress line every 2 minutes
            last_report = time.time()
            for gpu, (name, _) in running.items():
                print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu} {name}: {last_line(f'runs/{name}.log')}", flush=True)
        for gpu, (name, proc) in list(running.items()):
            if proc.poll() is not None:
                print(f"{name}: training process exited with code {proc.returncode}", flush=True)
                evaluate(name, gpu, False)
                del running[gpu]
    for n in args.names:                      # also evaluate runs that were already trained earlier
        if not args.dry_run and Path(f"runs/{n}/done.json").exists() and not Path(f"runs/{n}/results.json").exists():
            evaluate(n, 0, False)
    if not args.dry_run:
        summary()


if __name__ == "__main__":
    main()
