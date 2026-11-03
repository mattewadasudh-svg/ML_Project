#!/usr/bin/env python3
"""
make_splits.py -- attach the OFFICIAL FaceForensics++ train/val/test split to
the manifest written by `data_prep.py manifest`.

Official split files (splits/{train,val,test}.json) are lists of identity PAIRS,
e.g. ["071", "054"]. For a pair (a, b):
    original videos : a.mp4 and b.mp4
    fake videos     : a_b.mp4 and b_a.mp4   (for each of the 4 methods)
so every video of a pair, real or fake, goes to the pair's split. Because the
split is by pair, no identity appears in two splits (no identity leakage).

Expected result (paper / DeepfakeBench protocol):
    train 3600 videos (720 per method), val 700 (140), test 700 (140)

Usage:
  python make_splits.py --manifest crops/ffpp_manifest.csv --splits splits \
                        --out crops/ffpp_manifest_split.csv
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# forgery-type class index for the auxiliary head (0 = real). This 5-class
# choice is OUR assumption from the paper's Figure 1 -- see the deviation log.
FORGERY_CLASS = {"original": 0, "Deepfakes": 1, "Face2Face": 2,
                 "FaceSwap": 3, "NeuralTextures": 4}


def load_identity_split(split_dir):
    """identity id -> split name, from the official pair lists."""
    id2split = {}
    for name in ("train", "val", "test"):
        pairs = json.load(open(Path(split_dir) / f"{name}.json"))
        for a, b in pairs:
            for ident in (a, b):
                assert ident not in id2split, f"identity {ident} in two splits!"
                id2split[ident] = name
    return id2split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    id2split = load_identity_split(args.splits)
    rows = list(csv.DictReader(open(args.manifest)))
    out_rows, identities = [], {"train": set(), "val": set(), "test": set()}

    for r in rows:
        method, vid = r["method"], r["video_id"]
        ids = [vid] if method == "original" else vid.split("_")
        splits = {id2split[i] for i in ids}
        assert len(splits) == 1, f"{method}/{vid} spans splits {splits}"
        split = splits.pop()
        identities[split].update(ids)
        r["split"] = split
        r["forgery_type"] = FORGERY_CLASS[method]
        out_rows.append(r)

    # leakage check: identity sets must be disjoint
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (identities[a] & identities[b]), f"identity leak {a}/{b}"

    fields = ["dataset", "method", "video_id", "label", "forgery_type",
              "split", "n_frames", "path"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in out_rows)

    counts = Counter((r["split"], r["method"]) for r in out_rows)
    print(f"wrote {args.out} ({len(out_rows)} videos)\n")
    print(f"{'split':6s}" + "".join(f"{m:>16s}" for m in FORGERY_CLASS) + f"{'total':>8s}")
    expected = {"train": 720, "val": 140, "test": 140}
    ok = True
    for s in ("train", "val", "test"):
        line = f"{s:6s}"
        for m in FORGERY_CLASS:
            line += f"{counts[(s, m)]:16d}"
            ok &= counts[(s, m)] == expected[s]
        print(line + f"{sum(counts[(s, m)] for m in FORGERY_CLASS):8d}")
    nf = [int(r["n_frames"]) for r in out_rows]
    print(f"\nframes per video: min={min(nf)} max={max(nf)} total={sum(nf)}")
    print("matches official 720/140/140 per method:", "YES" if ok else "NO -- investigate")
    print("identity leakage between splits: none (asserted)")


if __name__ == "__main__":
    main()
