#!/usr/bin/env python3
"""
data_prep.py
Resumable face-crop preprocessing for DFF-Adapter.

Supported datasets:
    ffpp : FaceForensics++
    cdf1 : Celeb-DF v1
    cdf2 : Celeb-DF v2
    dfdc : DFDC train_sample_videos

For every video:
    1. Sample frames evenly throughout the video.
    2. Detect faces using MTCNN.
    3. Select the largest face in each frame.
    4. Fill missing detections using the nearest detected face.
    5. Crop and resize to the requested image size.
    6. Save the crops and a status file.

Output structure:
    crops/
        dfdc/
            REAL/
                video_id/
                    00000.jpg
                    ...
                    _status.json
            FAKE/
                video_id/
                    ...
        dfdc_manifest.csv
        dfdc_failed.txt
        dfdc_preview.png

Resumability:
    - Completed videos are skipped on subsequent runs.
    - Incomplete temporary folders are removed and retried.
    - Each video is marked complete only after its crops and status
      file have been written.

Thermal safety:
    --max_temp pauses processing if the NVIDIA GPU temperature reaches
    the configured threshold.
    --cooldown adds a delay between videos.
    Use nice -n 10 when launching to lower CPU scheduling priority.

Important:
    --max_temp monitors NVIDIA GPU temperature, not CPU temperature.
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


FFPP_FAKE = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]

REAL_METHODS = {
    "original",
    "Celeb-real",
    "YouTube-real",
    "REAL",
}


# ============================================================
# DATASET LISTING
# ============================================================

def list_ffpp(root):
    """
    Return (method, video_path) pairs for FaceForensics++.

    Videos are interleaved across manipulation methods so that
    a limited test run is not dominated by one method.
    """
    root = Path(root)

    per_method = {}

    real_dir = (
        root
        / "original_sequences"
        / "youtube"
        / "c23"
        / "videos"
    )

    per_method["original"] = sorted(real_dir.glob("*.mp4"))

    for method in FFPP_FAKE:
        video_dir = (
            root
            / "manipulated_sequences"
            / method
            / "c23"
            / "videos"
        )

        per_method[method] = sorted(video_dir.glob("*.mp4"))

    for method, videos in per_method.items():
        print(f"  {method:15s}: {len(videos)} videos")

    items = []

    max_count = max(
        (len(videos) for videos in per_method.values()),
        default=0,
    )

    for i in range(max_count):
        for method, videos in per_method.items():
            if i < len(videos):
                items.append((method, videos[i]))

    return items


def list_cdf2(root, list_file=None):
    """
    List Celeb-DF v1/v2 videos.

    If list_file is provided, use the specified video list.
    Official list entries may look like:

        1 YouTube-real/00170.mp4

    The final whitespace-separated token is treated as the
    relative video path.
    """
    root = Path(root)

    if list_file:
        list_file = Path(list_file)

        if not list_file.is_file():
            raise FileNotFoundError(
                f"Video list not found: {list_file}"
            )

        paths = []

        with open(list_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()

                if not line:
                    continue

                relative_path = line.split()[-1]
                paths.append(root / relative_path)

    else:
        paths = sorted(root.glob("*/*.mp4"))

    items = [
        (path.parent.name, path)
        for path in paths
        if path.is_file()
    ]

    missing = len(paths) - len(items)

    if missing:
        print(
            f"  WARNING: {missing} listed videos "
            f"were not found under {root}"
        )

    counts = {}

    for method, _ in items:
        counts[method] = counts.get(method, 0) + 1

    for method, count in sorted(counts.items()):
        print(f"  {method:15s}: {count} videos")

    return items


def list_dfdc(root):
    """
    List labeled videos from the DFDC training sample.

    Accepts either:

        Datasets/deepfake-detection-challenge/

    or:

        Datasets/deepfake-detection-challenge/train_sample_videos/

    Labels are read from metadata.json.

    metadata.json typically contains entries such as:

        {
            "video_name.mp4": {
                "label": "FAKE",
                "original": "another_video.mp4",
                "split": "train"
            }
        }

    Returns:
        [("REAL", video_path), ("FAKE", video_path), ...]

    REAL and FAKE videos are interleaved to make limited test runs
    more representative of both classes.
    """
    root = Path(root)

    if (root / "metadata.json").is_file():
        video_dir = root
    else:
        video_dir = root / "train_sample_videos"

    metadata_path = video_dir / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"DFDC metadata.json not found at {metadata_path}\n"
            "Pass either the DFDC dataset directory or the "
            "train_sample_videos directory."
        )

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Expected a JSON object in {metadata_path}"
        )

    real_items = []
    fake_items = []

    missing = 0
    unknown = 0

    for filename, info in metadata.items():

        if not isinstance(info, dict):
            unknown += 1
            continue

        label = str(
            info.get("label", "")
        ).strip().upper()

        if label not in {"REAL", "FAKE"}:
            unknown += 1
            continue

        video_path = video_dir / filename

        if not video_path.is_file():
            missing += 1
            continue

        item = (label, video_path)

        if label == "REAL":
            real_items.append(item)
        else:
            fake_items.append(item)

    real_items.sort(key=lambda item: item[1].name)
    fake_items.sort(key=lambda item: item[1].name)

    print(f"  REAL: {len(real_items)} videos")
    print(f"  FAKE: {len(fake_items)} videos")

    if missing:
        print(
            f"  WARNING: {missing} labeled videos "
            "are missing from disk"
        )

    if unknown:
        print(
            f"  WARNING: {unknown} metadata entries have "
            "unknown or missing labels"
        )

    items = []

    for i in range(max(len(real_items), len(fake_items))):

        if i < len(real_items):
            items.append(real_items[i])

        if i < len(fake_items):
            items.append(fake_items[i])

    return items


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def sample_frames(path, k):
    """
    Sample k approximately evenly spaced frames from a video.

    Returns:
        frames: list of BGR frames
        indices: corresponding original frame indices
    """
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        return [], []

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_frames <= 0:
        cap.release()
        return [], []

    indices = np.unique(
        np.linspace(
            0,
            n_frames - 1,
            k,
        ).round().astype(int)
    )

    wanted = set(indices.tolist())
    last = int(indices[-1])

    frames = []
    actual_indices = []

    i = 0

    while i <= last:

        if not cap.grab():
            break

        if i in wanted:
            ok, frame = cap.retrieve()

            if ok and frame is not None:
                frames.append(frame)
                actual_indices.append(i)

        i += 1

    cap.release()

    return frames, actual_indices


# ============================================================
# FACE DETECTION
# ============================================================

def detect_boxes(mtcnn, frames, det_max):
    """
    Detect the largest face in each frame.

    Detection is performed on resized copies to reduce computation.

    Returns:
        List of face boxes in original image coordinates.
        Each box is [x1, y1, x2, y2], or None if no face is detected.
    """
    if not frames:
        return []

    h, w = frames[0].shape[:2]

    scale_factor = min(
        1.0,
        det_max / max(h, w),
    )

    small_frames = []

    for frame in frames:

        if scale_factor < 1.0:
            resized = cv2.resize(
                frame,
                (
                    max(1, int(w * scale_factor)),
                    max(1, int(h * scale_factor)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = frame

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        small_frames.append(rgb)

    batch = np.stack(small_frames)

    boxes, _ = mtcnn.detect(batch)

    if boxes is None:
        return [None] * len(frames)

    results = []

    for box in boxes:

        if box is None or len(box) == 0:
            results.append(None)
            continue

        box = np.asarray(
            box,
            dtype=float,
        ).reshape(-1, 4)

        areas = (
            (box[:, 2] - box[:, 0])
            * (box[:, 3] - box[:, 1])
        )

        largest_index = int(np.argmax(areas))

        original_box = box[largest_index] / scale_factor

        results.append(original_box)

    return results


def fill_missing(boxes):
    """
    If a frame has no detected face, use the closest detected
    face box from another sampled frame.

    Returns None if no face was detected anywhere in the video.
    """
    found = [
        i
        for i, box in enumerate(boxes)
        if box is not None
    ]

    if not found:
        return None

    filled = list(boxes)

    for i in range(len(filled)):

        if filled[i] is None:

            nearest = min(
                found,
                key=lambda j: abs(j - i),
            )

            filled[i] = boxes[nearest]

    return filled


# ============================================================
# FACE CROPPING
# ============================================================

def crop_square(img, box, scale, size):
    """
    Crop a square centered on the face.

    Square side length:
        max(face_width, face_height) * scale

    The crop is resized to size x size.
    Regions outside the source image are zero-padded.
    """
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    side = max(
        x2 - x1,
        y2 - y1,
    ) * scale

    side = max(side, 1.0)

    s = size / side

    left = cx - side / 2
    top = cy - side / 2

    matrix = np.array(
        [
            [s, 0, -s * left],
            [0, s, -s * top],
        ],
        dtype=np.float32,
    )

    crop = cv2.warpAffine(
        img,
        matrix,
        (size, size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return crop


# ============================================================
# THERMAL MONITORING
# ============================================================

def gpu_temp():
    """
    Return the first NVIDIA GPU's temperature in Celsius.

    Returns None if nvidia-smi is unavailable or no temperature
    can be read.

    Note:
        This checks GPU temperature, not CPU temperature.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode != 0:
            return None

        values = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        if not values:
            return None

        temperatures = [
            int(value)
            for value in values
            if value.isdigit()
        ]

        if not temperatures:
            return None

        # Monitor the hottest visible NVIDIA GPU.
        return max(temperatures)

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        ValueError,
    ):
        return None


def wait_until_cool(max_temp, resume_margin=5):
    """
    Pause processing when GPU temperature reaches max_temp.

    Resume only when the GPU temperature falls to
    max_temp - resume_margin or below.

    If temperature monitoring is unavailable, continue and print
    a warning once rather than waiting forever.
    """
    if not max_temp or max_temp <= 0:
        return

    temperature = gpu_temp()

    if temperature is None:
        return

    if temperature < max_temp:
        return

    resume_temp = max_temp - resume_margin

    print(
        f"\nGPU temperature is {temperature} C "
        f"(limit {max_temp} C). Pausing."
    )

    while True:

        time.sleep(10)

        temperature = gpu_temp()

        if temperature is None:
            print(
                "WARNING: GPU temperature is no longer available. "
                "Resuming processing."
            )
            return

        print(
            f"GPU temperature: {temperature} C; "
            f"resume threshold: {resume_temp} C"
        )

        if temperature <= resume_temp:
            print("GPU has cooled sufficiently. Resuming.")
            return


# ============================================================
# CROPPING PIPELINE
# ============================================================

def run_crop(args):
    import torch
    from facenet_pytorch import MTCNN
    from tqdm import tqdm

    # ---------------- Dataset listing ----------------

    if args.dataset == "ffpp":

        print("Listing FaceForensics++ videos:")
        items = list_ffpp(args.root)

    elif args.dataset == "dfdc":

        print("Listing labeled DFDC training videos:")
        items = list_dfdc(args.root)

    else:

        print(f"Listing {args.dataset} videos:")
        items = list_cdf2(
            args.root,
            args.list,
        )

    if not items:
        sys.exit(
            "No videos found. Check --root and, if applicable, --list."
        )

    if args.limit > 0:
        items = items[:args.limit]

    print(f"\nVideos in queue: {len(items)}")

    # ---------------- Output setup ----------------

    out_root = Path(args.out)
    ds_out = out_root / args.dataset

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove leftover temporary directories for videos in this run.
    for method, _ in items:

        method_dir = ds_out / method

        if not method_dir.exists():
            continue

        for temp_dir in method_dir.glob(".tmp_*"):
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

    # ---------------- Device setup ----------------

    if torch.cuda.is_available():

        device = "cuda"

        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    else:

        device = "cpu"

        print(
            "WARNING: CUDA is unavailable. "
            "MTCNN will run on CPU and may be much slower."
        )

    print("Device:", device)

    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        thresholds=[0.6, 0.7, 0.7],
    )

    # ---------------- Logging ----------------

    failed_log = out_root / f"{args.dataset}_failed.txt"

    done = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    bar = tqdm(
        items,
        unit="video",
        dynamic_ncols=True,
    )

    # ---------------- Main processing loop ----------------

    try:

        for method, video_path in bar:

            # Pause before beginning the next video if GPU is hot.
            if args.max_temp > 0:
                wait_until_cool(args.max_temp)

            video_id = video_path.stem

            method_dir = ds_out / method

            final_dir = method_dir / video_id

            # Skip only completed videos.
            if (final_dir / "_status.json").is_file():
                skipped += 1

                bar.set_postfix(
                    done=done,
                    skipped=skipped,
                    failed=failed,
                )

                continue

            method_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Temporary directory makes the final rename atomic.
            temp_dir = method_dir / f".tmp_{video_id}"

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

            temp_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Sample video frames.
            frames, frame_indices = sample_frames(
                video_path,
                args.frames,
            )

            if not frames:

                with open(
                    failed_log,
                    "a",
                    encoding="utf-8",
                ) as fh:
                    fh.write(
                        f"{method}/{video_id}: "
                        "could not read video frames\n"
                    )

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True,
                )

                failed += 1

                bar.set_postfix(
                    done=done,
                    skipped=skipped,
                    failed=failed,
                )

                continue

            # Detect faces in small batches to limit GPU memory.
            boxes = []

            for start in range(
                0,
                len(frames),
                args.det_batch,
            ):

                frame_batch = frames[
                    start:start + args.det_batch
                ]

                batch_boxes = detect_boxes(
                    mtcnn,
                    frame_batch,
                    args.det_max,
                )

                boxes.extend(batch_boxes)

            n_missing = sum(
                box is None
                for box in boxes
            )

            filled_boxes = fill_missing(boxes)

            # No detectable face anywhere in this video.
            if filled_boxes is None:

                with open(
                    failed_log,
                    "a",
                    encoding="utf-8",
                ) as fh:
                    fh.write(
                        f"{method}/{video_id}: "
                        "no face detected\n"
                    )

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True,
                )

                failed += 1

                bar.set_postfix(
                    done=done,
                    skipped=skipped,
                    failed=failed,
                )

                continue

            # Save face crops.
            write_error = None
            saved_count = 0

            try:

                for frame, frame_index, box in zip(
                    frames,
                    frame_indices,
                    filled_boxes,
                ):

                    crop = crop_square(
                        frame,
                        box,
                        args.scale,
                        args.size,
                    )

                    crop_path = (
                        temp_dir
                        / f"{frame_index:05d}.jpg"
                    )

                    success = cv2.imwrite(
                        str(crop_path),
                        crop,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            95,
                        ],
                    )

                    if not success:
                        raise OSError(
                            f"Could not write {crop_path}"
                        )

                    saved_count += 1

                # Ensure all expected crops were written.
                if saved_count != len(frames):
                    raise OSError(
                        f"Saved {saved_count} crops, "
                        f"expected {len(frames)}"
                    )

                # Save completion metadata only after crops exist.
                status = {
                    "n_saved": saved_count,
                    "n_no_detection": int(n_missing),
                    "src": str(video_path),
                    "scale": args.scale,
                    "size": args.size,
                    "frames_requested": args.frames,
                    "dataset": args.dataset,
                    "method": method,
                    "video_id": video_id,
                }

                status_path = temp_dir / "_status.json"

                with open(
                    status_path,
                    "w",
                    encoding="utf-8",
                ) as fh:
                    json.dump(
                        status,
                        fh,
                        indent=2,
                    )

                # Flush files to disk before marking the video complete.
                if args.sync:

                    for file_path in temp_dir.iterdir():

                        if not file_path.is_file():
                            continue

                        with open(file_path, "rb") as fh:
                            os.fsync(fh.fileno())

                # If another completed output exists, do not overwrite it.
                if final_dir.exists():
                    shutil.rmtree(
                        final_dir,
                        ignore_errors=True,
                    )

                # Temporary and final folders share the same filesystem.
                os.rename(
                    temp_dir,
                    final_dir,
                )

                done += 1

            except Exception as exc:

                write_error = exc

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True,
                )

                with open(
                    failed_log,
                    "a",
                    encoding="utf-8",
                ) as fh:
                    fh.write(
                        f"{method}/{video_id}: "
                        f"{type(exc).__name__}: {exc}\n"
                    )

                failed += 1

            if write_error is not None:
                tqdm.write(
                    f"ERROR processing {method}/{video_id}: "
                    f"{write_error}"
                )

            elapsed_min = (
                time.time() - start_time
            ) / 60.0

            bar.set_postfix(
                done=done,
                skipped=skipped,
                failed=failed,
                elapsed_min=f"{elapsed_min:.1f}",
            )

            # Small delay between videos to reduce sustained load.
            if args.cooldown > 0:
                time.sleep(args.cooldown)

    except KeyboardInterrupt:

        print(
            "\nInterrupted. Completed videos have been retained. "
            "Run the same command again to resume."
        )

    finally:

        elapsed_min = (
            time.time() - start_time
        ) / 60.0

        print("\nPreprocessing summary")
        print("---------------------")
        print("Newly completed:", done)
        print("Already completed:", skipped)
        print("Failed:", failed)
        print(f"Elapsed time: {elapsed_min:.1f} minutes")
        print("Output directory:", ds_out)
        print("Failure log:", failed_log)


# ============================================================
# PREVIEW
# ============================================================

def run_preview(args):
    """
    Create a montage of saved face crops for visual inspection.
    """
    ds_out = Path(args.out) / args.dataset

    status_files = sorted(
        ds_out.glob("*/*/_status.json")
    )

    video_dirs = [
        path.parent
        for path in status_files
    ]

    if not video_dirs:
        sys.exit(
            f"No completed videos found in {ds_out}"
        )

    random.seed(0)

    selected = random.sample(
        video_dirs,
        min(len(video_dirs), args.preview_count),
    )

    tiles = []

    for video_dir in selected:

        jpgs = sorted(
            video_dir.glob("*.jpg")
        )

        if not jpgs:
            continue

        selected_jpg = random.choice(jpgs)

        image = cv2.imread(
            str(selected_jpg)
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            (160, 160),
        )

        label = (
            f"{video_dir.parent.name}/"
            f"{video_dir.name}"
        )[:24]

        cv2.putText(
            image,
            label,
            (3, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
        )

        tiles.append(image)

    if not tiles:
        sys.exit(
            "No readable JPEG crops found."
        )

    columns = 6

    while len(tiles) % columns:
        tiles.append(
            np.zeros_like(tiles[0])
        )

    rows = [
        np.hstack(
            tiles[i:i + columns]
        )
        for i in range(
            0,
            len(tiles),
            columns,
        )
    ]

    output_path = (
        Path(args.out)
        / f"{args.dataset}_preview.png"
    )

    cv2.imwrite(
        str(output_path),
        np.vstack(rows),
    )

    print("Preview written to:", output_path)


# ============================================================
# VERIFY
# ============================================================

def run_verify(args):
    """
    Verify recently completed videos.

    A video is considered invalid if:
        - Its status file cannot be read.
        - Its saved JPEG count differs from n_saved.
        - Any saved JPEG cannot be decoded.

    Invalid video folders are removed so a future crop run
    can regenerate them.
    """
    ds_out = Path(args.out) / args.dataset

    if not ds_out.exists():
        sys.exit(
            f"Dataset output directory does not exist: {ds_out}"
        )

    status_files = sorted(
        ds_out.glob("*/*/_status.json"),
        key=lambda path: path.stat().st_mtime,
    )

    total = len(status_files)

    if args.last > 0:
        status_files = status_files[-args.last:]

    bad = 0

    for status_path in status_files:

        video_dir = status_path.parent

        jpgs = sorted(
            video_dir.glob("*.jpg")
        )

        try:

            with open(
                status_path,
                "r",
                encoding="utf-8",
            ) as fh:
                metadata = json.load(fh)

            expected = int(
                metadata["n_saved"]
            )

            valid = (
                len(jpgs) == expected
                and expected > 0
            )

        except Exception:
            valid = False

        if valid:

            for jpg_path in jpgs:

                if cv2.imread(str(jpg_path)) is None:
                    valid = False
                    break

        if not valid:

            print(
                "Invalid video; removing:",
                video_dir,
            )

            shutil.rmtree(
                video_dir,
                ignore_errors=True,
            )

            bad += 1

    print(
        f"Checked {len(status_files)} of {total} videos; "
        f"removed {bad} invalid video folders."
    )


# ============================================================
# MANIFEST
# ============================================================

def run_manifest(args):
    """
    Create a CSV containing all completed videos.

    Label convention:
        0 = real
        1 = fake

    The DFDC REAL/FAKE labels are derived from the directory names,
    which are populated from metadata.json by list_dfdc().
    """
    ds_out = Path(args.out) / args.dataset

    status_files = sorted(
        ds_out.glob("*/*/_status.json")
    )

    rows = []

    for status_path in status_files:

        video_dir = status_path.parent

        try:

            with open(
                status_path,
                "r",
                encoding="utf-8",
            ) as fh:
                metadata = json.load(fh)

        except Exception:

            print(
                "WARNING: unreadable status file; skipping:",
                status_path,
            )

            continue

        method = video_dir.parent.name

        rows.append(
            {
                "dataset": args.dataset,
                "method": method,
                "video_id": video_dir.name,
                "label": (
                    0
                    if method in REAL_METHODS
                    else 1
                ),
                "n_frames": metadata.get("n_saved", 0),
                "path": str(
                    video_dir.relative_to(
                        Path(args.out)
                    )
                ),
            }
        )

    output_path = (
        Path(args.out)
        / f"{args.dataset}_manifest.csv"
    )

    # Explicit field names also handle an empty manifest safely.
    fieldnames = [
        "dataset",
        "method",
        "video_id",
        "label",
        "n_frames",
        "path",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter

    print(
        f"Wrote {output_path} "
        f"({len(rows)} completed videos)"
    )

    print(
        "Counts by method and label:",
        Counter(
            (row["method"], row["label"])
            for row in rows
        ),
    )


# ============================================================
# ARGUMENT PARSING
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Resumable MTCNN face-crop preprocessing "
            "for deepfake datasets."
        )
    )

    parser.add_argument(
        "mode",
        choices=[
            "crop",
            "preview",
            "manifest",
            "verify",
        ],
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "ffpp",
            "cdf1",
            "cdf2",
            "dfdc",
        ],
        required=True,
        help=(
            "Dataset name: ffpp, cdf1, cdf2, or dfdc"
        ),
    )

    parser.add_argument(
        "--root",
        help="Raw video dataset directory for crop mode",
    )

    parser.add_argument(
        "--out",
        default="crops",
        help="Output directory (default: crops)",
    )

    parser.add_argument(
        "--list",
        help="Optional official video list for Celeb-DF",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="Frames sampled per video (default: 32)",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output crop size (default: 224)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.3,
        help="Face crop expansion factor (default: 1.3)",
    )

    parser.add_argument(
        "--det_max",
        type=int,
        default=640,
        help=(
            "Maximum image dimension for face detection "
            "(default: 640)"
        ),
    )

    parser.add_argument(
        "--det_batch",
        type=int,
        default=8,
        help=(
            "Number of frames passed to MTCNN at once "
            "(default: 8)"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Process at most N videos; 0 means no limit"
        ),
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.25,
        help=(
            "Seconds to sleep between videos "
            "(default: 0.25)"
        ),
    )

    parser.add_argument(
        "--max_temp",
        type=int,
        default=78,
        help=(
            "Pause if GPU temperature reaches this "
            "value in Celsius; 0 disables the check"
        ),
    )

    parser.add_argument(
        "--last",
        type=int,
        default=50,
        help=(
            "Number of most recently completed videos "
            "to verify; 0 verifies all"
        ),
    )

    parser.add_argument(
        "--preview_count",
        type=int,
        default=24,
        help=(
            "Number of videos sampled for preview "
            "(default: 24)"
        ),
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Flush crop files to disk before marking "
            "each video complete"
        ),
    )

    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be at least 1")

    if args.size < 1:
        parser.error("--size must be at least 1")

    if args.det_batch < 1:
        parser.error("--det_batch must be at least 1")

    if args.det_max < 1:
        parser.error("--det_max must be at least 1")

    if args.scale <= 0:
        parser.error("--scale must be positive")

    if args.cooldown < 0:
        parser.error("--cooldown cannot be negative")

    if args.mode == "crop" and not args.root:
        parser.error("--root is required for crop mode")

    handlers = {
        "crop": run_crop,
        "preview": run_preview,
        "manifest": run_manifest,
        "verify": run_verify,
    }

    handlers[args.mode](args)


if __name__ == "__main__":
    main()