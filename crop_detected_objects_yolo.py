#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        ],
        key=lambda p: p.name.lower(),
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def format_float(value: float, precision: int = 6) -> str:
    text = f"{value:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def render_progress(current: int, total: int, name: str, start_time: float) -> None:
    if total <= 0:
        return
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(100 * current / total)
    eta_label = "ETA --:--"
    if current > 0:
        elapsed = time.monotonic() - start_time
        if elapsed > 0:
            rate = current / elapsed
            if rate > 0:
                remaining = max(0.0, (total - current) / rate)
                eta_time = datetime.now() + timedelta(seconds=remaining)
                eta_label = f"ETA {eta_time.strftime('%H:%M')}"

    tail = name
    if len(tail) > 40:
        tail = f"...{tail[-37:]}"
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} {percent:3d}% {eta_label} {tail}   "
    )
    sys.stdout.flush()


def parse_label_line(
    line: str, label_path: Path, line_no: int
) -> Optional[Tuple[str, List[float]]]:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) < 5:
        raise ValueError(
            f"Unexpected label format in {label_path} line {line_no}: "
            f"{len(parts)} tokens."
        )
    if len(parts) != 5 and (len(parts) - 5) % 3 != 0:
        raise ValueError(
            f"Unexpected label format in {label_path} line {line_no}: "
            f"{len(parts)} tokens."
        )
    class_id = parts[0]
    try:
        values = [float(x) for x in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric label values in {label_path} line {line_no}."
        ) from exc
    return class_id, values


def read_labels(label_path: Path) -> Tuple[List[Tuple[str, List[float]]], bool]:
    if not label_path.exists():
        return [], False
    lines = label_path.read_text(encoding="utf-8").splitlines()
    labels: List[Tuple[str, List[float]]] = []
    for idx, line in enumerate(lines, start=1):
        parsed = parse_label_line(line, label_path, idx)
        if parsed:
            labels.append(parsed)
    return labels, True


def collect_annotation_bounds(
    labels: Sequence[Tuple[str, List[float]]], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    found = False

    for _class_id, values in labels:
        if len(values) < 4:
            continue

        cx, cy, bw, bh = values[0], values[1], values[2], values[3]
        abs_cx = cx * width
        abs_cy = cy * height
        abs_w = bw * width
        abs_h = bh * height
        x1 = clamp(abs_cx - abs_w / 2.0, 0.0, float(width))
        y1 = clamp(abs_cy - abs_h / 2.0, 0.0, float(height))
        x2 = clamp(abs_cx + abs_w / 2.0, 0.0, float(width))
        y2 = clamp(abs_cy + abs_h / 2.0, 0.0, float(height))

        if x2 > x1 and y2 > y1:
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            found = True

        if len(values) > 4:
            keypoints = values[4:]
            for idx in range(0, len(keypoints), 3):
                if idx + 2 >= len(keypoints):
                    break
                kx, ky, vis = keypoints[idx], keypoints[idx + 1], keypoints[idx + 2]
                if vis <= 0 or (kx == 0.0 and ky == 0.0):
                    continue
                abs_kx = clamp(kx * width, 0.0, float(width))
                abs_ky = clamp(ky * height, 0.0, float(height))
                min_x = min(min_x, abs_kx)
                min_y = min(min_y, abs_ky)
                max_x = max(max_x, abs_kx)
                max_y = max(max_y, abs_ky)
                found = True

    if not found:
        return None
    return min_x, min_y, max_x, max_y


def build_crop(
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    margin_px: int,
) -> Tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = bounds
    left = int(max(0, min_x - margin_px))
    top = int(max(0, min_y - margin_px))
    right = int(min(width, max_x + margin_px))
    bottom = int(min(height, max_y + margin_px))

    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def transform_labels(
    labels: Sequence[Tuple[str, List[float]]],
    orig_w: int,
    orig_h: int,
    crop_box: Tuple[int, int, int, int],
) -> List[str]:
    left, top, right, bottom = crop_box
    crop_w = right - left
    crop_h = bottom - top
    out_lines: List[str] = []

    for class_id, values in labels:
        if len(values) < 4:
            continue
        cx, cy, bw, bh = values[0], values[1], values[2], values[3]
        abs_cx = cx * orig_w
        abs_cy = cy * orig_h
        abs_w = bw * orig_w
        abs_h = bh * orig_h

        x1 = clamp(abs_cx - abs_w / 2.0, 0.0, float(orig_w))
        y1 = clamp(abs_cy - abs_h / 2.0, 0.0, float(orig_h))
        x2 = clamp(abs_cx + abs_w / 2.0, 0.0, float(orig_w))
        y2 = clamp(abs_cy + abs_h / 2.0, 0.0, float(orig_h))

        new_x1 = clamp(x1 - left, 0.0, float(crop_w))
        new_y1 = clamp(y1 - top, 0.0, float(crop_h))
        new_x2 = clamp(x2 - left, 0.0, float(crop_w))
        new_y2 = clamp(y2 - top, 0.0, float(crop_h))

        new_bw = max(0.0, new_x2 - new_x1) / crop_w
        new_bh = max(0.0, new_y2 - new_y1) / crop_h
        if new_bw <= 0.0 or new_bh <= 0.0:
            continue
        new_cx = ((new_x1 + new_x2) / 2.0) / crop_w
        new_cy = ((new_y1 + new_y2) / 2.0) / crop_h

        out_values: List[float] = [clamp01(new_cx), clamp01(new_cy), clamp01(new_bw), clamp01(new_bh)]

        if len(values) > 4:
            keypoints = values[4:]
            for idx in range(0, len(keypoints), 3):
                if idx + 2 >= len(keypoints):
                    break
                kx = keypoints[idx]
                ky = keypoints[idx + 1]
                vis = keypoints[idx + 2]
                if vis <= 0 or (kx == 0.0 and ky == 0.0):
                    out_values.extend([0.0, 0.0, vis])
                    continue
                abs_kx = kx * orig_w
                abs_ky = ky * orig_h
                new_kx = clamp01((abs_kx - left) / crop_w)
                new_ky = clamp01((abs_ky - top) / crop_h)
                out_values.extend([new_kx, new_ky, vis])

        out_line = " ".join([class_id] + [format_float(v) for v in out_values])
        out_lines.append(out_line)

    return out_lines


def process(
    images_dir: Path,
    labels_dir: Path,
    images_out: Path,
    labels_out: Path,
    margin_px: int,
) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    labels_available = labels_dir.is_dir()
    if not labels_available:
        print(f"Warning: labels directory not found, labels will be skipped: {labels_dir}")

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    total = len(images)
    if total:
        print(f"Found {total} images. Processing...")
    else:
        print("No images found to process.")
        return

    start_time = time.monotonic()
    processed = 0
    cropped = 0
    copied_no_labels = 0
    missing_labels = 0
    empty_labels = 0
    dropped_labels = 0
    errors = 0

    for index, image_path in enumerate(images, start=1):
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                label_path = labels_dir / f"{image_path.stem}.txt"
                labels, has_label = read_labels(label_path) if labels_available else ([], False)

                if labels_available and not has_label:
                    missing_labels += 1
                if has_label and not labels:
                    empty_labels += 1

                crop_box = (0, 0, width, height)
                out_lines: List[str] = []

                if labels:
                    bounds = collect_annotation_bounds(labels, width, height)
                    if bounds:
                        crop_box = build_crop(bounds, width, height, margin_px)
                        if crop_box != (0, 0, width, height):
                            cropped += 1
                    out_lines = transform_labels(labels, width, height, crop_box)
                    dropped_labels += max(0, len(labels) - len(out_lines))
                else:
                    copied_no_labels += 1

                left, top, right, bottom = crop_box
                out_img = img.crop((left, top, right, bottom))
                out_img_path = images_out / image_path.name
                out_img.save(out_img_path)

                if has_label and out_lines:
                    out_label_path = labels_out / label_path.name
                    out_label_path.write_text(
                        "\n".join(out_lines) + "\n",
                        encoding="utf-8",
                    )
                processed += 1
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            errors += 1
            print(f"\nWarning: {image_path.name}: {exc}")
        finally:
            render_progress(index, total, image_path.name, start_time)

    if total:
        print()

    print("Done. Summary:")
    print(f"- Images processed: {processed}")
    print(f"- Images cropped: {cropped}")
    print(f"- Images without valid labels: {copied_no_labels}")
    print(f"- Missing label files: {missing_labels}")
    print(f"- Empty label files: {empty_labels}")
    print(f"- Dropped label lines after transform: {dropped_labels}")
    print(f"- Errors: {errors}")
    print(f"- Output images: {images_out}")
    print(f"- Output labels: {labels_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crop images around all YOLO detections/pose annotations with optional "
            "extra pixel boundary and rewrite labels."
        )
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Input images folder (default: images).",
    )
    parser.add_argument(
        "--labels-dir",
        default="labels",
        help="Input labels folder (default: labels).",
    )
    parser.add_argument(
        "--images-out",
        default="images-crop",
        help="Output images folder (default: images-crop).",
    )
    parser.add_argument(
        "--labels-out",
        default="labels-crop",
        help="Output labels folder (default: labels-crop).",
    )
    parser.add_argument(
        "--margin-px",
        type=int,
        default=20,
        help="Extra boundary around all annotations in pixels (default: 20).",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)
    margin_px = max(0, args.margin_px)

    print(
        "Cropping images to preserve all YOLO-detected objects/keypoints.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Images out: {images_out}\n"
        f"- Labels out: {labels_out}\n"
        f"- Extra boundary: {margin_px} px"
    )

    process(images_dir, labels_dir, images_out, labels_out, margin_px)


if __name__ == "__main__":
    main()
