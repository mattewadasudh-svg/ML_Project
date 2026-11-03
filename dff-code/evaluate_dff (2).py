"""
evaluate_dff.py -- video-level AUC / EER (the paper's protocol: frame scores are averaged per video).

  python evaluate_dff.py --ckpt runs/main/best.pt --backbone small \
      --sets /path/ff-crops/ffpp_manifest_split.csv:test /path/eval-corps/cdf1_manifest.csv:all \
             /path/eval-corps/cdf2_manifest.csv:all --out results.json
Each manifest sits at the root of its crops folder, so the crops root is taken from the manifest's directory.

AUC and EER are implemented in numpy (no scikit-learn needed); test_metrics.py checks them against sklearn.
torch is imported lazily so the metric functions can be tested anywhere.
"""
import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- metrics
def roc_auc(y_true, score):
    """Area under ROC = P(score of a random fake > score of a random real), ties count 1/2.
    Computed from ranks (Mann-Whitney U) with average ranks for ties."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
    _, start, count = np.unique(sorted_s, return_index=True, return_counts=True)
    for st, c in zip(start, count):
        if c > 1:
            ranks_sorted[st:st + c] = ranks_sorted[st:st + c].mean()
    ranks = np.empty(len(s))
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_points(y_true, score):
    """(fpr, tpr) at every distinct threshold, starting at (0, 0). Positive class = fake (1)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (1 + distinct) - tps
    return np.r_[0.0, fps / (len(y) - y.sum())], np.r_[0.0, tps / y.sum()]


def equal_error_rate(y_true, score):
    """EER: the operating point where false-positive rate == false-negative rate."""
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    fpr, tpr = roc_points(y, score)
    fnr = 1.0 - tpr
    d = fpr - fnr                      # increasing from -1 to +1
    j = int(np.argmax(d >= 0))
    if j == 0:
        return float(fpr[0])
    t = -d[j - 1] / (d[j] - d[j - 1])  # linear interpolation between the two neighbouring points
    return float(fpr[j - 1] + t * (fpr[j] - fpr[j - 1]))


def video_level(video_idx, frame_scores, video_labels):
    """Average frame scores per video. Returns (mean score per video, label per video) for videos seen."""
    n = len(video_labels)
    video_idx = np.asarray(video_idx)
    sums = np.bincount(video_idx, weights=np.asarray(frame_scores, float), minlength=n)
    cnt = np.bincount(video_idx, minlength=n)
    seen = cnt > 0
    return sums[seen] / cnt[seen], np.asarray(video_labels)[seen]


def summarize(video_idx, frame_scores, video_labels):
    mean, lab = video_level(video_idx, frame_scores, video_labels)
    frame_lab = np.asarray(video_labels)[np.asarray(video_idx)]
    return {"video_auc": roc_auc(lab, mean), "video_eer": equal_error_rate(lab, mean),
            "frame_auc": roc_auc(frame_lab, frame_scores),
            "n_videos": int(len(lab)), "n_frames": int(len(frame_scores))}


# --------------------------------------------------------------- inference
def run_inference(model, loader, device, amp=False):
    """P(fake) for every frame, task 0 only (inference uses the authenticity branch)."""
    import torch
    from dataset import gpu_normalize
    model.eval()
    vids, probs = [], []
    with torch.no_grad():
        for x, _, _, v in loader:
            x = gpu_normalize(x.to(device, non_blocking=True))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logit = model(x, task=0)
            probs.append(torch.sigmoid(logit.float()).cpu())
            vids.append(v)
    return torch.cat(vids).numpy(), torch.cat(probs).numpy()


def evaluate_dataset(model, dataset, device, amp=False, batch_size=64, num_workers=2):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    vids, probs = run_inference(model, loader, device, amp)
    return summarize(vids, probs, dataset.video_labels)


# --------------------------------------------------------------- CLI
def main():
    import torch
    from dataset import FrameDataset
    from train import build_model, load_checkpoint      # lazy: train.py imports this module
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="base")
    ap.add_argument("--variant", default="dff", choices=["dff", "lora", "linear"])
    ap.add_argument("--crops_root", default=None, help="only needed if a manifest path is relative")
    ap.add_argument("--sets", nargs="+", required=True, help="manifest.csv:split  (split 'all' = every video)")
    ap.add_argument("--frames", type=int, default=0, help="frames per video, 0 = all")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, variant=args.variant).to(device)
    load_checkpoint(model, args.ckpt)
    results = {}
    for spec in args.sets:
        manifest, split = spec.rsplit(":", 1)
        mpath = Path(manifest)
        if not mpath.is_absolute() and args.crops_root:
            mpath = Path(args.crops_root) / mpath
        ds = FrameDataset(str(mpath.parent), str(mpath),
                          None if split == "all" else split, mode="eval", frames_per_video=args.frames)
        r = evaluate_dataset(model, ds, device, amp=device.type == "cuda",
                             batch_size=args.batch_size, num_workers=args.num_workers)
        results[spec] = r
        print(f"{spec:40s} AUC={100 * r['video_auc']:.2f}  EER={100 * r['video_eer']:.2f}  "
              f"frameAUC={100 * r['frame_auc']:.2f}  ({r['n_videos']} videos, {r['n_frames']} frames)")
    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
