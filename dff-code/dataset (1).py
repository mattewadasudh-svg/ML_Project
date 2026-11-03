"""
dataset.py -- turns the cropped-frame folders + manifest CSV into training / evaluation samples.

Layout (made by data_prep.py and make_splits.py):
    <crops_root>/ffpp/<method>/<video_id>/<frame>.jpg        32 crops of 224x224 per video
    <crops_root>/ffpp_manifest_split.csv   columns: dataset,method,video_id,label,forgery_type,split,n_frames,path
    (cdf1 / cdf2 manifests have no 'split' column: use split=None = all videos)

Two modes:
  mode="train": one "epoch" = every video contributes `frames_per_video` RANDOM frames
                (this is how we cut the paper's compute: the paper trains on all 32 frames/video for 50 epochs)
  mode="eval" : deterministic. frames_per_video=0 -> all frames; k>0 -> k evenly spaced frames.
Every item returns: image (3,224,224 float, ImageNet-normalised), authenticity label (float 0/1),
forgery-type label (long 0..4), video index (long) -- the video index is used to average scores per video.
"""
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)   # each DataLoader worker is its own process: don't let OpenCV spawn extra threads

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet stats (what DINOv2 was trained with)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def gpu_normalize(x):
    """uint8 (B,3,H,W) on any device -> ImageNet-normalised float32. Identical numbers to the old CPU path,
    but the workers ship 4x fewer bytes and skip the float math."""
    if x.dtype != torch.uint8:
        return x
    m = torch.tensor(MEAN * 255.0, device=x.device).view(1, 3, 1, 1)
    sd = torch.tensor(STD * 255.0, device=x.device).view(1, 3, 1, 1)
    return (x.float() - m) / sd


def read_manifest(path, split=None):
    rows = list(csv.DictReader(open(path)))
    if split not in (None, "all"):
        rows = [r for r in rows if r.get("split") == split]
    if not rows:
        raise ValueError(f"no rows in {path} for split={split!r}")
    return rows


def build_index(vid_frames, mode, k):
    """vid_frames: list (one entry per video) of lists of frame paths.
    Returns the list of (video_index, frame_index_or_None) that makes up one pass over the data."""
    items = []
    for v, files in enumerate(vid_frames):
        if mode == "train":
            assert k > 0, "train mode needs frames_per_video > 0"
            items += [(v, None)] * k                          # frame chosen at random in __getitem__
        else:
            if k <= 0 or k >= len(files):
                idx = range(len(files))
            else:
                idx = np.unique(np.linspace(0, len(files) - 1, k).round().astype(int)).tolist()
            items += [(v, int(i)) for i in idx]
    return items


def augment(bgr):
    """Light augmentation as in the SOP: horizontal flip, brightness/contrast jitter, JPEG recompression."""
    if random.random() < 0.5:
        bgr = bgr[:, ::-1]
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)                      # contrast
        beta = random.uniform(-20, 20)                        # brightness
        bgr = np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        q = random.randint(60, 95)
        ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return bgr


class FrameDataset(Dataset):
    def __init__(self, crops_root, manifest, split=None, mode="eval", frames_per_video=0, augment_train=None, uint8=False):
        self.root = Path(crops_root)
        self.mode = mode
        self.uint8 = uint8          # True: return raw uint8 (normalise later with gpu_normalize)
        self.augment = (mode == "train") if augment_train is None else augment_train
        rows = read_manifest(manifest, split)
        self.vid_frames, kept = [], []
        for r in rows:
            d = self.root / r["path"]
            files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
            if not files:
                raise FileNotFoundError(f"no jpg frames in {d}")
            self.vid_frames.append([str(d / f) for f in files])
            kept.append(r)
        self.rows = kept
        self.video_labels = np.array([int(r["label"]) for r in kept])
        self.video_types = np.array([int(r.get("forgery_type", r["label"])) for r in kept])
        self.items = build_index(self.vid_frames, mode, frames_per_video)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v, f = self.items[i]
        files = self.vid_frames[v]
        path = files[random.randrange(len(files))] if f is None else files[f]
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"cannot read {path}")
        if self.augment:
            img = augment(img)
        img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
        if self.uint8:
            img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        else:
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(np.ascontiguousarray(((img - MEAN) / STD).transpose(2, 0, 1)))
        return (img,
                torch.tensor(float(self.video_labels[v]), dtype=torch.float32),
                torch.tensor(int(self.video_types[v]), dtype=torch.long),
                torch.tensor(v, dtype=torch.long))


# ------------------------------------------------- fake data for smoke tests
def make_fake_dataset(root, videos_per_split=None, frames=4, seed=0):
    """Tiny synthetic crops + manifest (brightness encodes real/fake, tint encodes forgery type)
    so the whole train/eval pipeline can be tested in seconds without the real data."""
    videos_per_split = videos_per_split or {"train": 30, "val": 10, "test": 10}
    methods = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    rng = np.random.default_rng(seed)
    root = Path(root)
    rows = []
    for split, n in videos_per_split.items():
        for i in range(n):
            t = i % 5
            vid = f"{split}{i:03d}"
            d = root / "ffpp" / methods[t] / vid
            d.mkdir(parents=True, exist_ok=True)
            for f in range(frames):
                img = np.full((224, 224, 3), 70 if t == 0 else 170, np.uint8)
                img[..., t % 3] = np.clip(img[..., t % 3] + 40 * (t // 3 + 1), 0, 255)
                img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
                cv2.imwrite(str(d / f"{f:05d}.jpg"), img)
            rows.append({"dataset": "ffpp", "method": methods[t], "video_id": vid,
                         "label": int(t != 0), "forgery_type": t, "split": split,
                         "n_frames": frames, "path": f"ffpp/{methods[t]}/{vid}"})
    manifest = root / "ffpp_manifest_split.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return str(manifest)
