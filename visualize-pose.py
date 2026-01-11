#!/usr/bin/env python3
"""Overlay YOLO pose annotations onto images and write results to disk."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
from tqdm import tqdm

# COCO-style 17 keypoints skeleton used by YOLO pose models.
DEFAULT_SKELETON: Sequence[Tuple[int, int]] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def parse_yolo_pose_line(
    line: str, img_w: int, img_h: int
) -> Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, float]]]:
    """Parse a single YOLO pose label line into class, box, and keypoints."""
    parts = line.strip().split()
    if len(parts) < 8:
        raise ValueError("Pose line too short to contain bounding box and keypoints.")

    class_id = int(float(parts[0]))
    cx, cy, bw, bh = map(float, parts[1:5])

    if (len(parts) - 5) % 3 != 0:
        raise ValueError("Pose line keypoints are not in triplets.")

    keypoints = []
    kp_triplets = (len(parts) - 5) // 3
    for i in range(kp_triplets):
        base = 5 + 3 * i
        x = float(parts[base]) * img_w
        y = float(parts[base + 1]) * img_h
        v = float(parts[base + 2])
        keypoints.append((x, y, v))

    # Convert box center/size to pixel values.
    box = (cx * img_w, cy * img_h, bw * img_w, bh * img_h)
    return class_id, box, keypoints


def load_annotations(
    label_path: Path,
    img_w: int,
    img_h: int,
    expected_keypoints: Optional[int],
) -> Tuple[List[Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, float]]]], int, int]:
    """Read a YOLO pose label file."""
    annotations = []
    skipped_lines = 0
    mismatched_keypoints = 0
    for idx, line in enumerate(label_path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            ann = parse_yolo_pose_line(line, img_w, img_h)
            annotations.append(ann)
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"Skipped line {idx + 1} in {label_path.name}: {exc}")
            skipped_lines += 1
            continue
        if expected_keypoints is not None and len(ann[2]) != expected_keypoints:
            mismatched_keypoints += 1
    return annotations, skipped_lines, mismatched_keypoints


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def load_expected_keypoints(dataset_yaml: Path, fallback: int = 17) -> int:
    if not dataset_yaml.is_file():
        print(f"Warning: dataset YAML not found at {dataset_yaml}; using {fallback} keypoints.")
        return fallback
    try:
        text = dataset_yaml.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not read dataset YAML {dataset_yaml}: {exc}; using {fallback}.")
        return fallback

    match = re.search(r"kpt_shape\s*:\s*\[\s*(\d+)\s*,", text)
    if not match:
        print(f"Warning: kpt_shape not found in {dataset_yaml}; using {fallback} keypoints.")
        return fallback
    return int(match.group(1))


def draw_pose(
    img: Image.Image,
    annotations,
    skeleton: Sequence[Tuple[int, int]] = DEFAULT_SKELETON,
    keypoint_radius: int = 4,
):
    """Draw keypoints, skeleton, and bounding boxes on the image."""
    draw = ImageDraw.Draw(img)
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (231, 76, 60),
        (155, 89, 182),
        (241, 196, 15),
        (26, 188, 156),
    ]

    for ann_idx, (class_id, box, keypoints) in enumerate(annotations):
        color = palette[ann_idx % len(palette)]
        cx, cy, bw, bh = box
        x0, y0 = cx - bw / 2, cy - bh / 2
        x1, y1 = cx + bw / 2, cy + bh / 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 4, y0 + 4), str(class_id), fill=color)

        # Draw skeleton lines where both endpoints are labeled (v > 0).
        for a, b in skeleton:
            if a >= len(keypoints) or b >= len(keypoints):
                continue
            xa, ya, va = keypoints[a]
            xb, yb, vb = keypoints[b]
            if va > 0 and vb > 0:
                draw.line([xa, ya, xb, yb], fill=color, width=3)

        # Draw keypoints.
        for x, y, v in keypoints:
            if v <= 0:
                continue
            draw.ellipse(
                [x - keypoint_radius, y - keypoint_radius, x + keypoint_radius, y + keypoint_radius],
                outline=color,
                width=3,
            )


def find_image_files(image_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in image_dir.iterdir():
        if path.suffix.lower() in exts and path.is_file():
            yield path


def visualize_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    expected_keypoints: Optional[int],
    keypoint_radius: int,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(find_image_files(images_dir))
    total_images = len(image_paths)
    if not total_images:
        print("No images found to visualize.")
        return {
            "total_images": 0,
            "images_with_labels": 0,
            "missing_labels": 0,
            "empty_annotations": 0,
            "saved_images": 0,
            "skipped_lines": 0,
            "mismatched_keypoints": 0,
            "image_errors": 0,
        }

    images_with_labels = 0
    missing_labels = 0
    empty_annotations = 0
    saved_images = 0
    skipped_lines_total = 0
    mismatched_keypoints_total = 0
    image_errors = 0

    for img_path in tqdm(image_paths, desc="Rendering", unit="image"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"Missing label for {img_path.name}; skipping.")
            missing_labels += 1
            continue

        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
        except OSError as exc:
            print(f"Failed to open image {img_path.name}: {exc}")
            image_errors += 1
            continue

        annotations, skipped_lines, mismatched_keypoints = load_annotations(
            label_path, img.width, img.height, expected_keypoints
        )
        skipped_lines_total += skipped_lines
        mismatched_keypoints_total += mismatched_keypoints
        images_with_labels += 1
        if not annotations:
            print(f"No annotations parsed for {img_path.name}; skipping.")
            empty_annotations += 1
            continue

        draw_pose(img, annotations, keypoint_radius=keypoint_radius)
        out_path = output_dir / img_path.name
        img.save(out_path)
        saved_images += 1

    return {
        "total_images": total_images,
        "images_with_labels": images_with_labels,
        "missing_labels": missing_labels,
        "empty_annotations": empty_annotations,
        "saved_images": saved_images,
        "skipped_lines": skipped_lines_total,
        "mismatched_keypoints": mismatched_keypoints_total,
        "image_errors": image_errors,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO11/YOLOv8 pose keypoints on images."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images"),
        help="Directory with source images.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels"),
        help="Directory with YOLO pose labels (.txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visual"),
        help="Directory to store rendered images.",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("dataset.yaml"),
        help="Dataset YAML file containing kpt_shape.",
    )
    parser.add_argument(
        "--keypoint-radius",
        type=int,
        default=4,
        help="Radius of keypoint circles in pixels.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(args.images, cwd)
    labels_dir = resolve_path(args.labels, cwd)
    output_dir = resolve_path(args.output, cwd)
    dataset_yaml = resolve_path(args.dataset_yaml, cwd)
    expected_keypoints = load_expected_keypoints(dataset_yaml)

    print(
        "Visualizing YOLO pose labels.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Output dir: {output_dir}\n"
        f"- Dataset YAML: {dataset_yaml}\n"
        f"- Expected keypoints: {expected_keypoints}\n"
        f"- Keypoint radius: {args.keypoint_radius}"
    )

    if not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    if not labels_dir.is_dir():
        print(f"Error: Labels directory not found: {labels_dir}")
        return 1

    stats = visualize_dataset(
        images_dir,
        labels_dir,
        output_dir,
        expected_keypoints,
        args.keypoint_radius,
    )
    print(
        "Done.\n"
        f"- Images found: {stats['total_images']}\n"
        f"- Images with labels: {stats['images_with_labels']}\n"
        f"- Missing labels: {stats['missing_labels']}\n"
        f"- Empty annotations: {stats['empty_annotations']}\n"
        f"- Saved images: {stats['saved_images']}\n"
        f"- Skipped label lines: {stats['skipped_lines']}\n"
        f"- Keypoint count mismatches: {stats['mismatched_keypoints']}\n"
        f"- Image read errors: {stats['image_errors']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
