#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

DEFAULT_COCO_SKELETON = [
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [6, 12],
    [7, 13],
    [12, 13],
    [12, 14],
    [14, 16],
    [13, 15],
    [15, 17],
]


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def resolve_config_path(path: Path, cwd: Path, script_dir: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = cwd / path
    if cwd_candidate.exists():
        return cwd_candidate
    return script_dir / path


def render_progress(current: int, total: int, name: str, start_time: float) -> None:
    if total <= 0:
        return
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(100 * current / total)
    tail = name
    if len(tail) > 40:
        tail = f"...{tail[-37:]}"
    eta_label = "ETA --:--"
    if current > 0:
        elapsed = time.monotonic() - start_time
        if elapsed > 0:
            rate = current / elapsed
            if rate > 0:
                remaining = max(0.0, (total - current) / rate)
                eta_time = datetime.now() + timedelta(seconds=remaining)
                eta_label = f"ETA {eta_time.strftime('%H:%M')}"
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} {percent:3d}% {eta_label} {tail}   "
    )
    sys.stdout.flush()


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        ],
        key=lambda p: p.name.lower(),
    )


def parse_dataset_yaml(dataset_yaml: Path) -> Tuple[int, List[str], Dict[int, str]]:
    kpt_count = 17
    kpt_names: List[str] = []
    class_names: Dict[int, str] = {}

    if not dataset_yaml.is_file():
        return kpt_count, kpt_names, class_names

    try:
        lines = dataset_yaml.read_text(encoding="utf-8").splitlines()
    except OSError:
        return kpt_count, kpt_names, class_names

    in_names_block = False
    in_kpt_names_block = False
    in_kpt_names_class0 = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("kpt_shape"):
            parts = stripped.split("[", 1)
            if len(parts) > 1:
                nums = parts[1].split("]")[0].split(",")
                if nums:
                    try:
                        kpt_count = int(nums[0].strip())
                    except ValueError:
                        pass

        if stripped.startswith("names:"):
            in_names_block = True
            inline = stripped.split(":", 1)[1].strip()
            if inline.startswith("[") and inline.endswith("]"):
                items = inline[1:-1].split(",")
                for idx, item in enumerate(items):
                    name = item.strip().strip("'\"")
                    if name:
                        class_names[idx] = name
                in_names_block = False
            continue

        if stripped.startswith("kpt_names:"):
            in_kpt_names_block = True
            in_kpt_names_class0 = False
            continue

        if in_names_block:
            if line.lstrip() == line:
                in_names_block = False
            else:
                if ":" in stripped:
                    idx_text, name = stripped.split(":", 1)
                    try:
                        idx = int(idx_text.strip())
                    except ValueError:
                        continue
                    name = name.strip().strip("'\"")
                    if name:
                        class_names[idx] = name
            continue

        if in_kpt_names_block:
            if line.lstrip() == line:
                in_kpt_names_block = False
                in_kpt_names_class0 = False
                continue
            if stripped.endswith(":"):
                idx_text = stripped[:-1].strip()
                try:
                    idx = int(idx_text)
                except ValueError:
                    in_kpt_names_class0 = False
                    continue
                in_kpt_names_class0 = idx == 0
                continue
            if in_kpt_names_class0 and stripped.startswith("- "):
                name = stripped[2:].strip().strip("'\"")
                if name:
                    kpt_names.append(name)

    return kpt_count, kpt_names, class_names


def yolo_vis_to_coco(vis: float, threshold: float) -> int:
    if not math.isfinite(vis) or vis <= 0:
        return 0
    if vis <= 1:
        return 2 if vis >= threshold else 0
    vis_int = int(round(vis))
    return max(0, min(2, vis_int))


def yolo_bbox_to_coco(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[List[float], float] | None:
    if len(bbox) < 4:
        return None
    cx, cy, bw, bh = bbox[:4]
    abs_w = bw * img_w
    abs_h = bh * img_h
    if abs_w <= 0 or abs_h <= 0:
        return None
    x0 = (cx * img_w) - abs_w / 2.0
    y0 = (cy * img_h) - abs_h / 2.0
    x1 = x0 + abs_w
    y1 = y0 + abs_h
    x0 = max(0.0, min(float(img_w), x0))
    y0 = max(0.0, min(float(img_h), y0))
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    area = w * h
    return [x0, y0, w, h], area


def parse_label_line(
    line: str, label_path: Path, line_no: int, expected_kpts: int | None
) -> Tuple[int, List[float], List[Tuple[float, float, float]]]:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(
            f"Unexpected label format in {label_path} line {line_no}: "
            f"{len(parts)} tokens."
        )
    try:
        class_id = int(float(parts[0]))
    except ValueError as exc:
        raise ValueError(
            f"Invalid class id in {label_path} line {line_no}."
        ) from exc
    try:
        values = [float(x) for x in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric label values in {label_path} line {line_no}."
        ) from exc
    bbox = values[:4]
    kpt_vals = values[4:]
    if kpt_vals and len(kpt_vals) % 3 != 0:
        raise ValueError(
            f"Keypoints not in triplets in {label_path} line {line_no}."
        )
    kpts = []
    for idx in range(0, len(kpt_vals), 3):
        if idx + 2 >= len(kpt_vals):
            break
        kpts.append((kpt_vals[idx], kpt_vals[idx + 1], kpt_vals[idx + 2]))

    if expected_kpts is not None and expected_kpts > 0:
        if len(kpts) < expected_kpts:
            kpts.extend([(0.0, 0.0, 0.0)] * (expected_kpts - len(kpts)))
        elif len(kpts) > expected_kpts:
            kpts = kpts[:expected_kpts]

    return class_id, bbox, kpts


def build_categories(
    class_names: Dict[int, str], used_classes: List[int], kpt_names: List[str]
) -> List[Dict]:
    categories: List[Dict] = []
    if not class_names:
        class_names = {0: "person"}
    for class_id in sorted(used_classes):
        name = class_names.get(class_id, f"class_{class_id}")
        supercategory = "person" if name == "person" else "object"
        category = {
            "id": class_id + 1,
            "name": name,
            "supercategory": supercategory,
        }
        if kpt_names:
            category["keypoints"] = kpt_names
            if len(kpt_names) == 17:
                category["skeleton"] = DEFAULT_COCO_SKELETON
        categories.append(category)
    return categories


def process_split(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    kpt_names: List[str],
    kpt_count: int,
    class_names: Dict[int, str],
    vis_threshold: float,
) -> Dict[str, int]:
    images_split = images_dir / split
    labels_split = labels_dir / split

    image_paths = list_images(images_split)
    label_paths = (
        sorted(labels_split.glob("*.txt")) if labels_split.is_dir() else []
    )
    label_stems = {path.stem for path in label_paths}
    image_stems = {path.stem for path in image_paths}

    stats = {
        "total_images": len(image_paths),
        "images_with_labels": 0,
        "missing_labels": 0,
        "labels_without_images": len(label_stems - image_stems),
        "empty_labels": 0,
        "annotations": 0,
        "skipped_lines": 0,
        "image_errors": 0,
        "label_errors": 0,
    }

    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    used_classes: List[int] = []

    image_id = 1
    annotation_id = 1
    start_time = time.monotonic()

    if not image_paths:
        print(f"No images found for split '{split}'.")
    for idx, image_path in enumerate(image_paths, start=1):
        render_progress(idx, len(image_paths), image_path.name, start_time)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except (UnidentifiedImageError, OSError) as exc:
            print(f"\nSkipping image {image_path.name}: {exc}")
            stats["image_errors"] += 1
            continue

        rel_name = image_path.relative_to(images_dir).as_posix()
        coco_images.append(
            {
                "id": image_id,
                "file_name": rel_name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_split / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["missing_labels"] += 1
            image_id += 1
            continue

        stats["images_with_labels"] += 1
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            print(f"\nSkipping label {label_path.name}: {exc}")
            stats["label_errors"] += 1
            image_id += 1
            continue

        line_annotations = 0
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                class_id, bbox, kpts = parse_label_line(
                    line, label_path, line_no, kpt_count
                )
            except ValueError as exc:
                print(f"\nSkipped line {line_no} in {label_path.name}: {exc}")
                stats["skipped_lines"] += 1
                continue

            coco_bbox = yolo_bbox_to_coco(bbox, width, height)
            if coco_bbox is None:
                stats["skipped_lines"] += 1
                continue
            bbox_out, area = coco_bbox

            keypoints: List[float] = []
            num_keypoints = 0
            for kp_x, kp_y, kp_v in kpts:
                abs_x = kp_x * width
                abs_y = kp_y * height
                v = yolo_vis_to_coco(kp_v, vis_threshold)
                if v > 0:
                    num_keypoints += 1
                keypoints.extend([abs_x, abs_y, v])

            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox_out,
                    "area": area,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_keypoints,
                }
            )
            annotation_id += 1
            line_annotations += 1
            if class_id not in used_classes:
                used_classes.append(class_id)

        if line_annotations == 0:
            stats["empty_labels"] += 1

        stats["annotations"] += line_annotations
        image_id += 1

    if image_paths:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if not kpt_names and kpt_count == 17:
        kpt_names = DEFAULT_COCO_KEYPOINTS[:]
    if kpt_names and len(kpt_names) != kpt_count:
        kpt_names = [f"kp_{idx}" for idx in range(kpt_count)]

    categories = build_categories(class_names, used_classes, kpt_names)

    coco = {
        "info": {
            "description": "YOLO pose to COCO conversion",
            "version": "1.0",
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"coco_{split}.json"
    output_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert YOLO11-pose labels to COCO JSON annotations."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Images folder containing train/val subfolders (default: images).",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="labels",
        help="Labels folder containing train/val subfolders (default: labels).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated dataset splits to convert (default: train,val).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="coco",
        help="Output folder for COCO JSON files (default: coco).",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default="dataset.yaml",
        help="Dataset YAML with keypoint/class names (default: dataset.yaml).",
    )
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=0.5,
        help="Visibility threshold for 0-1 keypoint values (default: 0.5).",
    )

    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    output_dir = resolve_path(Path(args.output_dir), cwd)
    dataset_yaml = resolve_config_path(Path(args.dataset_yaml), cwd, script_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    kpt_count, kpt_names, class_names = parse_dataset_yaml(dataset_yaml)

    print(
        "Converting YOLO11-pose labels to COCO JSON.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Output dir: {output_dir}\n"
        f"- Splits: {', '.join(splits)}\n"
        f"- Dataset YAML: {dataset_yaml}\n"
        f"- Keypoints: {kpt_count}\n"
        f"- Visibility threshold: {args.vis_threshold}"
    )

    if not splits:
        print("Error: No dataset splits provided.")
        return 1
    if not images_dir.is_dir():
        print(f"Error: Images directory '{images_dir}' not found.")
        return 1
    if not labels_dir.is_dir():
        print(f"Error: Labels directory '{labels_dir}' not found.")
        return 1

    totals = {
        "total_images": 0,
        "images_with_labels": 0,
        "missing_labels": 0,
        "labels_without_images": 0,
        "empty_labels": 0,
        "annotations": 0,
        "skipped_lines": 0,
        "image_errors": 0,
        "label_errors": 0,
    }

    for split in splits:
        print(f"\nProcessing split: {split}")
        stats = process_split(
            split=split,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            kpt_names=kpt_names,
            kpt_count=kpt_count,
            class_names=class_names,
            vis_threshold=args.vis_threshold,
        )
        for key in totals:
            totals[key] += stats.get(key, 0)

        print(
            "Split stats:\n"
            f"- Images: {stats['total_images']}\n"
            f"- Images with labels: {stats['images_with_labels']}\n"
            f"- Missing labels: {stats['missing_labels']}\n"
            f"- Labels without images: {stats['labels_without_images']}\n"
            f"- Empty labels: {stats['empty_labels']}\n"
            f"- Annotations: {stats['annotations']}\n"
            f"- Skipped lines: {stats['skipped_lines']}\n"
            f"- Image errors: {stats['image_errors']}\n"
            f"- Label errors: {stats['label_errors']}"
        )

    print(
        "\nFinished COCO conversion.\n"
        "Totals:\n"
        f"- Images: {totals['total_images']}\n"
        f"- Images with labels: {totals['images_with_labels']}\n"
        f"- Missing labels: {totals['missing_labels']}\n"
        f"- Labels without images: {totals['labels_without_images']}\n"
        f"- Empty labels: {totals['empty_labels']}\n"
        f"- Annotations: {totals['annotations']}\n"
        f"- Skipped lines: {totals['skipped_lines']}\n"
        f"- Image errors: {totals['image_errors']}\n"
        f"- Label errors: {totals['label_errors']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
