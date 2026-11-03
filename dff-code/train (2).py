"""
train.py -- dual-supervision training of the DFF-Adapter on a frozen DINOv2.

Per batch (paper, 'Training Details'): the SAME images go through the network twice,
    pass 1 (task 0): authenticity logit  -> BCE-with-logits loss        L_bce   (Eq. 7)
    pass 2 (task 1): forgery-type logits -> cross-entropy loss          L_ftc   (Eq. 9)
    loss = lambda0 * L_bce + lambda1 * L_ftc   (Eq. 10; paper: 10 and 2), one backward pass.
Only adapters + the two linear heads update; the DINOv2 weights never change.

Built to survive Kaggle/Colab session limits:
  * checkpoint every --ckpt_every steps and at every epoch end (atomic write: temp file then rename)
  * on startup, resumes from <out_dir>/last.pt if it exists (also mid-epoch)
  * --max_hours: saves and exits cleanly before the session limit  (prints TIME_LIMIT_REACHED)
  * checkpoints hold only the trainable weights (a few MB), so handing over to a teammate = copying last.pt

Examples
  python train.py --smoke                                         # 1-2 min pipeline test on fake data
  python train.py --backbone base --crops_root /kaggle/working/crops --out_dir runs/base_main \
        --epochs 10 --frames_per_video 8 --batch_size 24 --max_hours 3
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from dataset import FrameDataset, make_fake_dataset, gpu_normalize
from dff_adapter import DFFConfig, DFFDetector
from evaluate_dff import evaluate_dataset

HF_NAMES = {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large"}   # NOTE: the paper uses the *with-registers* large model


VARIANT_KIND = {"dff": "dff", "lora": "lora", "linear": "none"}


def build_model(name, n_classes=5, variant="dff"):
    from transformers import AutoModel, Dinov2Config, Dinov2Model
    if name == "tiny":   # random 2-layer model, only for smoke tests
        cfg = Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                           mlp_ratio=4, image_size=224, patch_size=14)
        backbone = Dinov2Model(cfg).eval()
    else:
        try:
            backbone = AutoModel.from_pretrained(HF_NAMES[name], attn_implementation="sdpa")
        except Exception:
            backbone = AutoModel.from_pretrained(HF_NAMES[name])
    return DFFDetector(backbone, DFFConfig(), n_classes, kind=VARIANT_KIND[variant])


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)           # rename is atomic: a crash never leaves a half-written last.pt


def load_checkpoint(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:3]}"
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert set(missing) <= frozen, "checkpoint is missing trainable weights"
    return ck


class EpochSampler(Sampler):
    """Reproducible shuffle for (seed, epoch); `start` lets a resumed run skip batches already done."""
    def __init__(self, n, seed, epoch, start=0):
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        self.idx = torch.randperm(n, generator=g)[start:].tolist()

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="base", choices=["tiny", "small", "base", "large"])
    ap.add_argument("--crops_root", default="crops")
    ap.add_argument("--manifest", default=None, help="default: <crops_root>/ffpp_manifest_split.csv")
    ap.add_argument("--out_dir", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--frames_per_video", type=int, default=8, help="random frames per video per epoch")
    ap.add_argument("--val_frames", type=int, default=8, help="evenly spaced frames per val video")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--lambda_auth", type=float, default=10.0)
    ap.add_argument("--lambda_type", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="forgery-type loss only (our extension)")
    ap.add_argument("--no_aux", action="store_true", help="ablation: skip the forgery-type pass")
    ap.add_argument("--variant", default="dff", choices=["dff", "lora", "linear"],
                    help="dff = ours | lora = plain LoRA baseline | linear = frozen backbone, train heads only "
                         "(baselines are single-task, so --no_aux is implied)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_hours", type=float, default=0, help="stop cleanly after this many hours (0 = no limit)")
    ap.add_argument("--ckpt_every", type=int, default=300, help="steps between checkpoints")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="fast pipeline test on fake data")
    args = ap.parse_args()
    if args.variant != "dff":
        args.no_aux = True

    sys.stdout.reconfigure(line_buffering=True)   # progress lines appear immediately, even when redirected to a file
    t_start = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.backbone, args.epochs, args.frames_per_video, args.val_frames = "tiny", 2, 2, 2
        args.batch_size, args.num_workers, args.log_every, args.ckpt_every = 8, 0, 2, 3
        args.crops_root = str(out / "smoke_data")
        args.manifest = make_fake_dataset(args.crops_root)
    manifest = args.manifest or f"{args.crops_root}/ffpp_manifest_split.csv"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"             # fp16 autocast + GradScaler (T4 has no fast bf16)

    train_ds = FrameDataset(args.crops_root, manifest, "train", "train", args.frames_per_video, uint8=True)
    val_ds = FrameDataset(args.crops_root, manifest, "val", "eval", args.val_frames, uint8=True)
    print(f"train: {len(train_ds.vid_frames)} videos -> {len(train_ds)} samples/epoch | "
          f"val: {len(val_ds.vid_frames)} videos, {len(val_ds)} frames | device={device} amp={amp}")

    model = build_model(args.backbone, variant=args.variant).to(device)

    n_gpus = torch.cuda.device_count()
    train_model = model
    if n_gpus > 1:
        train_model = torch.nn.DataParallel(
            model,
            device_ids=list(range(n_gpus)),
            output_device=0
        )
        print(f"Multi-GPU training enabled: {n_gpus} GPUs", flush=True)
    else:
        print("Single-GPU training", flush=True)
    params = model.trainable_parameters()
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epoch, batches_done, global_step, best_auc = 0, 0, 0, -1.0
    last = out / "last.pt"
    if last.exists():
        ck = load_checkpoint(model, last)
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        epoch, batches_done, global_step, best_auc = ck["epoch"], ck["batches_done"], ck["step"], ck["best_auc"]
        print(f"RESUMED from {last}: epoch={epoch} batch={batches_done} step={global_step} best_auc={best_auc:.4f}")
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    def checkpoint(epoch_, batches_, tag="last.pt"):
        save_atomic({"model": trainable_state(model), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch_, "batches_done": batches_, "step": global_step, "best_auc": best_auc,
                     "args": vars(args)}, out / tag)

    def log(row):
        row["time_min"] = round((time.time() - t_start) / 60, 2)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    print(f"plan: {args.epochs} epochs x {steps_per_epoch} steps = {total_steps} steps "
          f"(batch {args.batch_size}, {len(train_ds)} samples/epoch)")
    window = {"la": 0.0, "lt": 0.0, "acc": 0.0, "n": 0}
    win_t0 = time.time()
    for ep in range(epoch, args.epochs):
        sampler = EpochSampler(len(train_ds), args.seed, ep, start=batches_done * args.batch_size)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=amp,
                            prefetch_factor=4 if args.num_workers > 0 else None)
        model.train()
        t_wait = time.time()
        for b, (x, y_auth, y_type, _) in enumerate(loader, start=batches_done):
            if b == batches_done:   # how long did the data pipeline take to deliver its first batch?
                print(f"epoch {ep}: first batch ready after {time.time() - t_wait:.1f}s", flush=True)
                win_t0 = time.time()
            x = gpu_normalize(x.to(device, non_blocking=True))
            y_auth, y_type = y_auth.to(device, non_blocking=True), y_type.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logit = train_model(x, task=0)                                           # pass 1: authenticity
                l_auth = F.binary_cross_entropy_with_logits(logit.float(), y_auth)
                if args.no_aux:
                    l_type, acc = torch.zeros((), device=device), float("nan")
                    loss = args.lambda_auth * l_auth
                else:
                    type_logits = train_model(x, task=1).float()                         # pass 2: forgery type
                    l_type = F.cross_entropy(type_logits, y_type, label_smoothing=args.label_smoothing)
                    acc = (type_logits.argmax(1) == y_type).float().mean()   # stays on GPU: no sync
                    loss = args.lambda_auth * l_auth + args.lambda_type * l_type
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            window["la"] += l_auth.detach(); window["lt"] += l_type.detach(); window["acc"] += acc; window["n"] += 1   # no .item(): no per-step GPU sync
            if global_step % args.log_every == 0:
                n = window["n"]
                speed = n * args.batch_size / max(time.time() - win_t0, 1e-9)           # samples / second
                eta_min = (total_steps - global_step) * args.batch_size / max(speed, 1e-9) / 60
                row = {"epoch": ep, "step": global_step, "of": total_steps, "loss_auth": round(float(window["la"]) / n, 4),
                       "loss_type": round(float(window["lt"]) / n, 4), "acc_type": round(float(window["acc"]) / n, 3),
                       "samples_s": round(speed, 1), "eta_min": round(eta_min, 1)}
                print(row, flush=True)
                log(row)
                window = {"la": 0.0, "lt": 0.0, "acc": 0.0, "n": 0}
                win_t0 = time.time()
            if global_step % args.ckpt_every == 0:
                checkpoint(ep, b + 1)
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                checkpoint(ep, b + 1)
                print(f"TIME_LIMIT_REACHED at epoch={ep} batch={b + 1} step={global_step}; saved {last}")
                return
        batches_done = 0
        res = evaluate_dataset(model, val_ds, device, amp, batch_size=args.batch_size * 2,
                               num_workers=args.num_workers)
        print(f"== epoch {ep} validation: video AUC={res['video_auc']:.4f} EER={res['video_eer']:.4f} "
              f"frame AUC={res['frame_auc']:.4f}")
        log({"epoch": ep, "step": global_step, **{f"val_{k}": v for k, v in res.items()}})
        if res["video_auc"] > best_auc:
            best_auc = res["video_auc"]
            checkpoint(ep + 1, 0, "best.pt")
        checkpoint(ep + 1, 0)
    json.dump({"best_val_auc": best_auc, "steps": global_step}, open(out / "done.json", "w"))
    print(f"DONE. best val video AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
