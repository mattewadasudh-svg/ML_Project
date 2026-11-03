"""bench.py -- is training data-bound or GPU-bound?  Run once on Kaggle before (re)starting training.
python bench.py --crops_root $FF_ROOT --manifest $FF_MANIFEST --num_workers 4
Prints samples/s for (1) the data loader alone and (2) the GPU step alone on a fixed batch.
Training speed ~= min(1, 2). If 1 < 2 you are data-bound; otherwise GPU-bound."""
import argparse, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FrameDataset, gpu_normalize
from train import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--crops_root", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--batch_size", type=int, default=24); ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--backbone", default="base", choices=["small", "base", "large"])
a = ap.parse_args()

ds = FrameDataset(a.crops_root, a.manifest, "train", "train", 16, uint8=True)
dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, pin_memory=True,
                prefetch_factor=4 if a.num_workers else None)
it = iter(dl); next(it)
t = time.time()
for _ in range(a.n): next(it)
data_sps = a.n * a.batch_size / (time.time() - t)

dev = torch.device("cuda")
model = build_model(a.backbone).to(dev).train()
opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
x8 = torch.randint(0, 255, (a.batch_size, 3, 224, 224), dtype=torch.uint8, device=dev)
ya = torch.randint(0, 2, (a.batch_size,), device=dev).float(); yt = torch.randint(0, 5, (a.batch_size,), device=dev)
def step():
    x = gpu_normalize(x8)
    with torch.autocast("cuda", dtype=torch.float16):
        l = 10 * F.binary_cross_entropy_with_logits(model(x, task=0).float(), ya) + 2 * F.cross_entropy(model(x, task=1).float(), yt)
    opt.zero_grad(set_to_none=True); scaler.scale(l).backward(); scaler.step(opt); scaler.update()
for _ in range(3): step()
torch.cuda.synchronize(); t = time.time()
for _ in range(a.n): step()
torch.cuda.synchronize()
gpu_sps = a.n * a.batch_size / (time.time() - t)
print(f"data loader alone : {data_sps:6.1f} samples/s  (workers={a.num_workers})")
print(f"GPU step alone    : {gpu_sps:6.1f} samples/s")
print("=> DATA-BOUND (fix I/O / workers)" if data_sps < gpu_sps * 0.9 else "=> GPU-BOUND (reduce compute: see options)")
