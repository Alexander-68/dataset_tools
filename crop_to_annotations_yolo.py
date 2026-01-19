#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

PREFIX_BY_RATIO = {
    (1, 1): "11_",
    (3, 4): "34_",
    (4, 3): "43_",
    (1, 2): "12_",
    (2, 1): "21_",
    (2, 3): "23_",
    (3, 2): "32_",
}


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def list_images(source_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        ],
        key=lambda p: p.name.lower(),
    )


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


def format_float(value: float, precision: int = 6) -> str:
    text = f"{value:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def format_vis(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return format_float(value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


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
    if len(parts) == 5:
        expected = 5
    elif (len(parts) - 5) % 3 == 0:
        expected = len(parts)
    else:
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
    if len(parts) != expected:
        raise ValueError(
            f"Unexpected label length in {label_path} line {line_no}."
        )
    return class_id, values


def read_labels(label_path: Path) -> Tuple[List[Tuple[str, List[float]]], List[str], bool]:
    if not label_path.exists():
        return [], [], False
    lines = label_path.read_text(encoding="utf-8").splitlines()
    raw_lines: List[str] = []
    labels: List[Tuple[str, List[float]]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        raw_lines.append(stripped)
        parsed = parse_label_line(stripped, label_path, idx)
        if parsed:
            labels.append(parsed)
    return labels, raw_lines, True


def collect_geometry(
    labels: Iterable[Tuple[str, List[float]]], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    has_data = False

    for _class_id, values in labels:
        if len(values) < 4:
            continue
        cx, cy, bw, bh = values[0], values[1], values[2], values[3]
        abs_cx = cx * width
        abs_cy = cy * height
        abs_w = bw * width
        abs_h = bh * height
        x1 = abs_cx - abs_w / 2.0
        y1 = abs_cy - abs_h / 2.0
        x2 = abs_cx + abs_w / 2.0
        y2 = abs_cy + abs_h / 2.0
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1 = clamp(x1, 0.0, float(width))
        x2 = clamp(x2, 0.0, float(width))
        y1 = clamp(y1, 0.0, float(height))
        y2 = clamp(y2, 0.0, float(height))
        if x2 > x1 and y2 > y1:
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            has_data = True

        if len(values) > 4:
            keypoints = values[4:]
            for idx in range(0, len(keypoints), 3):
                if idx + 2 >= len(keypoints):
                    break
                x, y, v = keypoints[idx], keypoints[idx + 1], keypoints[idx + 2]
                if v <= 0 or (x == 0.0 and y == 0.0):
                    continue
                abs_x = clamp(x * width, 0.0, float(width))
                abs_y = clamp(y * height, 0.0, float(height))
                min_x = min(min_x, abs_x)
                min_y = min(min_y, abs_y)
                max_x = max(max_x, abs_x)
                max_y = max(max_y, abs_y)
                has_data = True

    if not has_data:
        return None
    return min_x, min_y, max_x, max_y


def add_padding(
    rect: Tuple[float, float, float, float],
    padding: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    if padding <= 0:
        return rect
    min_x, min_y, max_x, max_y = rect
    rect_w = max_x - min_x
    rect_h = max_y - min_y
    if rect_w <= 0 or rect_h <= 0:
        return rect
    pad_w = rect_w * padding
    pad_h = rect_h * padding
    return (
        clamp(min_x - pad_w, 0.0, float(width)),
        clamp(min_y - pad_h, 0.0, float(height)),
        clamp(max_x + pad_w, 0.0, float(width)),
        clamp(max_y + pad_h, 0.0, float(height)),
    )


def fit_crop(
    rect: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    target_w: float,
    target_h: float,
) -> Optional[Tuple[int, int, int, int]]:
    if target_w <= 0 or target_h <= 0:
        return None
    if target_w > img_w or target_h > img_h:
        return None

    min_x, min_y, max_x, max_y = rect

    left_min = max(0.0, max_x - target_w)
    left_max = min(min_x, img_w - target_w)
    top_min = max(0.0, max_y - target_h)
    top_max = min(min_y, img_h - target_h)

    if left_min > left_max or top_min > top_max:
        return None

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    desired_left = center_x - target_w / 2.0
    desired_top = center_y - target_h / 2.0

    left = clamp(desired_left, left_min, left_max)
    top = clamp(desired_top, top_min, top_max)

    right = left + target_w
    bottom = top + target_h

    left_i = int(math.floor(left))
    top_i = int(math.floor(top))
    right_i = int(math.ceil(right))
    bottom_i = int(math.ceil(bottom))

    left_i = int(clamp(left_i, 0, img_w - 1))
    top_i = int(clamp(top_i, 0, img_h - 1))
    right_i = int(clamp(right_i, left_i + 1, img_w))
    bottom_i = int(clamp(bottom_i, top_i + 1, img_h))

    return left_i, top_i, right_i, bottom_i


def fit_ratio_crop(
    rect: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    ratio_w: int,
    ratio_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    rect_w = rect[2] - rect[0]
    rect_h = rect[3] - rect[1]
    if rect_w <= 0 or rect_h <= 0:
        return None

    ratio = ratio_w / ratio_h
    target_w = max(rect_w, rect_h * ratio)
    target_h = target_w / ratio
    if target_h < rect_h:
        target_h = rect_h
        target_w = target_h * ratio

    return fit_crop(rect, img_w, img_h, target_w, target_h)


def fit_largest_square(
    rect: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    rect_w = rect[2] - rect[0]
    rect_h = rect[3] - rect[1]
    side = min(img_w, img_h)
    if rect_w > side or rect_h > side:
        return None
    return fit_crop(rect, img_w, img_h, float(side), float(side))


def crop_area(crop: Tuple[int, int, int, int]) -> int:
    return max(0, crop[2] - crop[0]) * max(0, crop[3] - crop[1])


def normalize_ratio_crop(
    crop: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    ratio_w: int,
    ratio_h: int,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = crop
    width = right - left
    height = bottom - top
    if width <= 1 or height <= 1:
        return crop
    if ratio_w <= 0 or ratio_h <= 0:
        return crop
    unit = min(width / ratio_w, height / ratio_h)
    unit_int = int(math.floor(unit))
    if unit_int % 2 == 1:
        unit_int -= 1
    if unit_int < 1:
        return crop
    new_w = unit_int * ratio_w
    new_h = unit_int * ratio_h
    if new_w <= 0 or new_h <= 0:
        return crop
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    new_left = int(round(center_x - new_w / 2.0))
    new_top = int(round(center_y - new_h / 2.0))
    new_left = int(clamp(new_left, 0, img_w - new_w))
    new_top = int(clamp(new_top, 0, img_h - new_h))
    return new_left, new_top, new_left + new_w, new_top + new_h


def pick_crop(
    rect: Tuple[float, float, float, float],
    padded_rect: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
    square_crop = fit_ratio_crop(padded_rect, img_w, img_h, 1, 1)
    if square_crop:
        return square_crop, (1, 1)

    largest_square = fit_largest_square(rect, img_w, img_h)
    if largest_square:
        return largest_square, (1, 1)

    ratio_pairs = [
        [(1, 2), (2, 1)],
        [(3, 4), (4, 3)],
        [(2, 3), (3, 2)],
    ]
    for pair in ratio_pairs:
        candidates: List[Tuple[Tuple[int, int, int, int], Tuple[int, int]]] = []
        for ratio in pair:
            crop = fit_ratio_crop(padded_rect, img_w, img_h, ratio[0], ratio[1])
            if crop:
                candidates.append((crop, ratio))
        if candidates:
            candidates.sort(key=lambda item: crop_area(item[0]))
            return candidates[0]

    return None, None


def transform_labels(
    labels: Iterable[Tuple[str, List[float]]],
    width: int,
    height: int,
    crop: Tuple[int, int, int, int],
) -> List[str]:
    left, top, right, bottom = crop
    crop_w = right - left
    crop_h = bottom - top
    out_lines: List[str] = []

    for class_id, values in labels:
        if len(values) < 4:
            continue
        cx, cy, bw, bh = values[0], values[1], values[2], values[3]
        abs_cx = cx * width
        abs_cy = cy * height
        abs_w = bw * width
        abs_h = bh * height

        new_cx = (abs_cx - left) / crop_w
        new_cy = (abs_cy - top) / crop_h
        new_w = abs_w / crop_w
        new_h = abs_h / crop_h

        parts = [
            class_id,
            format_float(clamp01(new_cx)),
            format_float(clamp01(new_cy)),
            format_float(clamp01(new_w)),
            format_float(clamp01(new_h)),
        ]

        if len(values) > 4:
            keypoints = values[4:]
            for idx in range(0, len(keypoints), 3):
                if idx + 2 >= len(keypoints):
                    break
                x, y, v = keypoints[idx], keypoints[idx + 1], keypoints[idx + 2]
                if v <= 0 or (x == 0.0 and y == 0.0):
                    parts.extend(["0", "0", "0"])
                    continue
                abs_x = x * width
                abs_y = y * height
                new_x = (abs_x - left) / crop_w
                new_y = (abs_y - top) / crop_h
                parts.append(format_float(clamp01(new_x)))
                parts.append(format_float(clamp01(new_y)))
                parts.append(format_vis(v))

        out_lines.append(" ".join(parts))

    return out_lines


def resize_to_max(img: Image.Image, max_dim: Optional[int]) -> Tuple[Image.Image, bool]:
    if not max_dim or max_dim <= 0:
        return img, False
    width, height = img.size
    max_side = max(width, height)
    if max_side <= max_dim:
        return img, False
    scale = max_dim / max_side
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS), True


def process_images(
    images_dir: Path,
    labels_dir: Path,
    images_out: Path,
    labels_out: Path,
    padding_percent: float,
    max_dim: Optional[int],
) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    labels_available = labels_dir.is_dir()
    if not labels_available:
        print(f"Warning: Labels folder not found: {labels_dir}. No cropping will occur.")

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    files = list_images(images_dir)
    total = len(files)
    if total:
        print(f"Found {total} images. Processing...")
    else:
        print("No images found to process.")
        return

    padding = max(0.0, padding_percent / 100.0)
    start_time = time.monotonic()

    processed = 0
    cropped = 0
    kept = 0
    resized = 0
    missing_labels = 0
    empty_labels = 0
    skipped_empty = 0
    errors = 0
    prefix_counts: dict[str, int] = {}

    for index, image_path in enumerate(files, start=1):
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size

                labels: List[Tuple[str, List[float]]] = []
                raw_lines: List[str] = []
                has_label_file = False
                if labels_available:
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    try:
                        labels, raw_lines, has_label_file = read_labels(label_path)
                    except ValueError as exc:
                        print(f"Warning: {exc}")
                        errors += 1
                        labels = []
                        raw_lines = []
                        has_label_file = False

                if labels_available and not has_label_file:
                    missing_labels += 1
                if has_label_file and not labels:
                    empty_labels += 1
                    skipped_empty += 1
                    continue

                crop = None
                ratio_used: Optional[Tuple[int, int]] = None
                if labels:
                    rect = collect_geometry(labels, width, height)
                    if rect:
                        padded_rect = add_padding(rect, padding, width, height)
                        crop, ratio_used = pick_crop(rect, padded_rect, width, height)
                        if crop and ratio_used in {(1, 1), (1, 2), (2, 1)}:
                            crop = normalize_ratio_crop(
                                crop,
                                width,
                                height,
                                ratio_used[0],
                                ratio_used[1],
                            )

                if crop:
                    cropped += 1
                    crop_box = crop
                    cropped_img = img.crop(crop_box)
                else:
                    kept += 1
                    crop_box = (0, 0, width, height)
                    cropped_img = img

                resized_img, did_resize = resize_to_max(cropped_img, max_dim)
                if did_resize:
                    resized += 1

                ratio_key = ratio_used if ratio_used else None
                prefix = PREFIX_BY_RATIO.get(ratio_key, "ff_")

                output_name = f"{prefix}{image_path.stem}.jpg"
                output_path = images_out / output_name

                save_img = resized_img
                if save_img.mode not in ("RGB", "L"):
                    save_img = save_img.convert("RGB")
                save_img.save(output_path, format="JPEG", quality=95)

                if has_label_file:
                    if crop:
                        out_lines = transform_labels(labels, width, height, crop_box)
                    else:
                        out_lines = raw_lines
                    out_label_path = labels_out / f"{prefix}{image_path.stem}.txt"
                    out_label_path.write_text(
                        "\n".join(out_lines) + ("\n" if out_lines else ""),
                        encoding="utf-8",
                    )

                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
                processed += 1
        except (UnidentifiedImageError, OSError):
            errors += 1
        finally:
            render_progress(index, total, image_path.name, start_time)

    if total:
        print()

    print("Done. Summary:")
    print(f"- Images processed: {processed}")
    print(f"- Cropped: {cropped}")
    print(f"- Kept original: {kept}")
    print(f"- Resized: {resized}")
    print(f"- Missing labels: {missing_labels}")
    print(f"- Empty labels: {empty_labels}")
    print(f"- Skipped empty labels: {skipped_empty}")
    print(f"- Errors: {errors}")
    print(f"- Output images: {images_out}")
    print(f"- Output labels: {labels_out}")
    if prefix_counts:
        ordered = [
            "11_",
            "34_",
            "43_",
            "23_",
            "32_",
            "12_",
            "21_",
            "ff_",
        ]
        counts = [f"{key}{prefix_counts.get(key, 0)}" for key in ordered]
        print(f"- Prefix counts: {', '.join(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crop images and YOLO annotations to fit all objects and keypoints."
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
        "--padding",
        type=float,
        default=20.0,
        help="Padding percent around annotations (default: 20).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1024,
        help="Max output image dimension in pixels (default: 1024). Use 0 to disable.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)

    print(
        "Cropping images to fit YOLO annotations.\n"
        f"- CWD: {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Images out: {images_out}\n"
        f"- Labels out: {labels_out}\n"
        f"- Padding: {args.padding:.2f}%\n"
        f"- Max dimension: {args.max_dim}"
    )

    process_images(
        images_dir,
        labels_dir,
        images_out,
        labels_out,
        args.padding,
        args.max_dim,
    )


if __name__ == "__main__":
    main()
