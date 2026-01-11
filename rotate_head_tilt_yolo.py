#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
KPT_COUNT = 17
POSE_VALUE_COUNT = 4 + KPT_COUNT * 3
PERSON_CLASS = "0"
NOSE_IDX = 0
LEFT_EYE_IDX = 1
RIGHT_EYE_IDX = 2

ROTATE_MIN_DEG = 10.0
ROTATE_MAX_DEG = 40.0


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate portrait images around the nose when the eye line tilt is "
            "below 15 degrees, updating YOLO11-pose annotations."
        )
    )
    parser.add_argument("--images-dir", default="images", help="Input images folder.")
    parser.add_argument("--labels-dir", default="labels", help="Input labels folder.")
    parser.add_argument(
        "--images-out", default="images-r", help="Output images folder."
    )
    parser.add_argument(
        "--labels-out", default="labels-r", help="Output labels folder."
    )
    parser.add_argument(
        "--no-crop",
        dest="crop",
        action="store_false",
        help="Keep the expanded canvas with black borders.",
    )
    parser.add_argument(
        "--tilt-threshold",
        type=float,
        default=15.0,
        help="Rotate only if eye-line tilt is below this many degrees.",
    )
    parser.add_argument(
        "--rotate-min",
        type=float,
        default=ROTATE_MIN_DEG,
        help="Minimum clockwise rotation angle in degrees.",
    )
    parser.add_argument(
        "--rotate-max",
        type=float,
        default=ROTATE_MAX_DEG,
        help="Maximum clockwise rotation angle in degrees.",
    )
    parser.add_argument(
        "--name-prefix",
        default="r_",
        help="Prefix to add to rotated output filenames.",
    )
    parser.set_defaults(crop=True)
    return parser.parse_args()


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ],
        key=lambda p: p.name.lower(),
    )


def parse_label_line(
    line: str, label_path: Path, line_no: int
) -> Optional[Tuple[str, List[float]]]:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) == 5:
        expected = 5
    elif len(parts) == 1 + POSE_VALUE_COUNT:
        expected = 1 + POSE_VALUE_COUNT
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


def read_labels(label_path: Path) -> List[Tuple[str, List[float]]]:
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    labels: List[Tuple[str, List[float]]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = parse_label_line(stripped, label_path, idx)
        if parsed:
            labels.append(parsed)
    return labels


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


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def render_progress(current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def rotate_point(
    x: float, y: float, cos_a: float, sin_a: float, cx: float, cy: float
) -> Tuple[float, float]:
    dx = x - cx
    dy = y - cy
    return (
        cx + cos_a * dx - sin_a * dy,
        cy + sin_a * dx + cos_a * dy,
    )


def rotation_params(
    width: int, height: int, center: Tuple[float, float], angle_rad: float
) -> Tuple[float, float, float, float, int, int]:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = center
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    rotated = [rotate_point(x, y, cos_a, sin_a, cx, cy) for x, y in corners]
    min_x = min(x for x, _ in rotated)
    max_x = max(x for x, _ in rotated)
    min_y = min(y for _, y in rotated)
    max_y = max(y for _, y in rotated)
    min_x_floor = math.floor(min_x)
    min_y_floor = math.floor(min_y)
    max_x_ceil = math.ceil(max_x)
    max_y_ceil = math.ceil(max_y)
    out_w = int(max_x_ceil - min_x_floor)
    out_h = int(max_y_ceil - min_y_floor)
    offset_x = -min_x_floor
    offset_y = -min_y_floor
    return cos_a, sin_a, offset_x, offset_y, out_w, out_h


def affine_inverse(
    cos_a: float,
    sin_a: float,
    cx: float,
    cy: float,
    offset_x: float,
    offset_y: float,
) -> Tuple[float, float, float, float, float, float]:
    a = cos_a
    b = sin_a
    c = cx - cos_a * (cx + offset_x) - sin_a * (cy + offset_y)
    d = -sin_a
    e = cos_a
    f = cy + sin_a * (cx + offset_x) - cos_a * (cy + offset_y)
    return (a, b, c, d, e, f)


def max_inscribed_rect(
    width: int, height: int, angle_rad: float
) -> Tuple[float, float]:
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    angle = abs(angle_rad) % math.pi
    if angle > math.pi / 2:
        angle = math.pi - angle
    if angle < 1e-6:
        return float(width), float(height)
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    if width <= 2 * sin_a * cos_a * height or height <= 2 * sin_a * cos_a * width:
        x = 0.5 * min(width, height)
        if width < height:
            return x / sin_a, x / cos_a
        return x / cos_a, x / sin_a
    cos_2a = cos_a * cos_a - sin_a * sin_a
    return (
        (width * cos_a - height * sin_a) / cos_2a,
        (height * cos_a - width * sin_a) / cos_2a,
    )


def transform_point(
    x: float,
    y: float,
    cos_a: float,
    sin_a: float,
    center: Tuple[float, float],
    offset: Tuple[float, float],
    crop_offset: Tuple[float, float],
) -> Tuple[float, float]:
    cx, cy = center
    rot_x, rot_y = rotate_point(x, y, cos_a, sin_a, cx, cy)
    out_x = rot_x + offset[0] - crop_offset[0]
    out_y = rot_y + offset[1] - crop_offset[1]
    return out_x, out_y


def find_largest_pose(
    labels: Iterable[Tuple[str, List[float]]]
) -> Optional[List[float]]:
    best = None
    best_area = -1.0
    for class_id, values in labels:
        if class_id != PERSON_CLASS:
            continue
        if len(values) < POSE_VALUE_COUNT:
            continue
        area = values[2] * values[3]
        if area > best_area:
            best_area = area
            best = values
    return best


def get_keypoint(values: List[float], index: int) -> Optional[Tuple[float, float, float]]:
    start = 4 + index * 3
    if len(values) < start + 3:
        return None
    return values[start], values[start + 1], values[start + 2]


def keypoint_valid(kpt: Tuple[float, float, float]) -> bool:
    x, y, v = kpt
    if v <= 0:
        return False
    if x == 0.0 and y == 0.0:
        return False
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return False
    return True


def compute_tilt_deg(
    left_eye: Tuple[float, float, float],
    right_eye: Tuple[float, float, float],
    width: int,
    height: int,
) -> float:
    dx = abs((right_eye[0] - left_eye[0]) * width)
    dy = abs((right_eye[1] - left_eye[1]) * height)
    if dx == 0 and dy == 0:
        return 90.0
    return abs(math.degrees(math.atan2(dy, dx)))


def transform_labels(
    labels: List[Tuple[str, List[float]]],
    img_w: int,
    img_h: int,
    cos_a: float,
    sin_a: float,
    center: Tuple[float, float],
    offset: Tuple[float, float],
    crop_offset: Tuple[float, float],
    out_w: int,
    out_h: int,
) -> List[str]:
    out_lines: List[str] = []
    for class_id, values in labels:
        if len(values) < 4:
            continue
        cx = values[0] * img_w
        cy = values[1] * img_h
        half_w = (values[2] * img_w) / 2.0
        half_h = (values[3] * img_h) / 2.0
        corners = [
            (cx - half_w, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
        rotated = [
            transform_point(x, y, cos_a, sin_a, center, offset, crop_offset)
            for x, y in corners
        ]
        xs = [pt[0] for pt in rotated]
        ys = [pt[1] for pt in rotated]
        min_x = max(0.0, min(xs))
        max_x = min(float(out_w), max(xs))
        min_y = max(0.0, min(ys))
        max_y = min(float(out_h), max(ys))
        if max_x <= min_x or max_y <= min_y:
            continue
        bbox_cx = (min_x + max_x) / 2.0
        bbox_cy = (min_y + max_y) / 2.0
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        parts = [
            class_id,
            format_float(clamp01(bbox_cx / out_w)),
            format_float(clamp01(bbox_cy / out_h)),
            format_float(clamp01(bbox_w / out_w)),
            format_float(clamp01(bbox_h / out_h)),
        ]
        if len(values) > 4:
            kps = values[4:]
            for idx in range(0, len(kps), 3):
                x, y, v = kps[idx], kps[idx + 1], kps[idx + 2]
                if v <= 0 or (x == 0.0 and y == 0.0):
                    parts.extend(["0", "0", "0"])
                    continue
                abs_x = x * img_w
                abs_y = y * img_h
                out_x, out_y = transform_point(
                    abs_x, abs_y, cos_a, sin_a, center, offset, crop_offset
                )
                if out_x < 0 or out_x > out_w or out_y < 0 or out_y > out_h:
                    parts.extend(["0", "0", "0"])
                else:
                    parts.extend(
                        [
                            format_float(clamp01(out_x / out_w)),
                            format_float(clamp01(out_y / out_h)),
                            format_vis(v),
                        ]
                    )
        out_lines.append(" ".join(parts))
    return out_lines


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)

    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(images_dir)
    if not image_paths:
        print("No images found.")
        return 1

    print("Rotate head tilt in portraits (YOLO11-pose).")
    print(f"Images: {images_dir} -> {images_out}")
    print(f"Labels: {labels_dir} -> {labels_out}")
    print(f"Crop: {args.crop}")
    print(f"Output prefix: {args.name_prefix!r}")
    print(
        "Tilt threshold: "
        f"{args.tilt_threshold} deg, rotate clockwise: "
        f"{args.rotate_min}-{args.rotate_max} deg"
    )
    print("Skipping output for images that do not rotate.")
    print(f"Total images: {len(image_paths)}")

    rng = random.Random()
    rotated_images = 0
    kept_images = 0
    rotated_labels = 0
    kept_labels = 0

    for idx, path in enumerate(image_paths, start=1):
        label_path = labels_dir / f"{path.stem}.txt"
        labels = read_labels(label_path)
        best = find_largest_pose(labels)
        rotate = False
        angle_deg = 0.0
        nose_kpt: Optional[Tuple[float, float, float]] = None
        if best:
            left_eye = get_keypoint(best, LEFT_EYE_IDX)
            right_eye = get_keypoint(best, RIGHT_EYE_IDX)
            nose_kpt = get_keypoint(best, NOSE_IDX)
            if left_eye and right_eye and nose_kpt:
                if keypoint_valid(left_eye) and keypoint_valid(right_eye) and keypoint_valid(nose_kpt):
                    with Image.open(path) as probe:
                        width, height = probe.size
                    tilt = compute_tilt_deg(left_eye, right_eye, width, height)
                    if tilt < args.tilt_threshold:
                        rotate = True
                        angle_deg = -rng.uniform(args.rotate_min, args.rotate_max)

        if not rotate:
            kept_images += 1
            if label_path.exists():
                kept_labels += 1
            render_progress(idx, len(image_paths))
            continue

        if nose_kpt is None:
            kept_images += 1
            if label_path.exists():
                kept_labels += 1
            render_progress(idx, len(image_paths))
            continue

        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            nose_x = nose_kpt[0] * width
            nose_y = nose_kpt[1] * height
            angle_rad = math.radians(angle_deg)
            center = (nose_x, nose_y)
            cos_a, sin_a, offset_x, offset_y, out_w, out_h = rotation_params(
                width, height, center, angle_rad
            )
            affine = affine_inverse(cos_a, sin_a, nose_x, nose_y, offset_x, offset_y)
            rotated = img.transform(
                (out_w, out_h),
                Image.AFFINE,
                affine,
                resample=Image.BICUBIC,
                fillcolor=(0, 0, 0),
            )

            crop_left = 0
            crop_top = 0
            if args.crop:
                added_w = max(0, out_w - width)
                added_h = max(0, out_h - height)
                crop_x = int(round(added_w / 3.0))
                crop_y = int(round(added_h / 2.0))
                crop_x = max(0, min(crop_x, (out_w - 1) // 2))
                crop_y = max(0, min(crop_y, (out_h - 1) // 2))
                crop_left = crop_x
                crop_right = crop_x
                crop_top = crop_y
                crop_bottom = crop_y
                crop_w = out_w - crop_left - crop_right
                crop_h = out_h - crop_top - crop_bottom
                rotated = rotated.crop(
                    (crop_left, crop_top, crop_left + crop_w, crop_top + crop_h)
                )
                out_w, out_h = crop_w, crop_h

            out_lines = transform_labels(
                labels,
                width,
                height,
                cos_a,
                sin_a,
                center,
                (offset_x, offset_y),
                (float(crop_left), float(crop_top)),
                out_w,
                out_h,
            )

        out_img_path = images_out / f"{args.name_prefix}{path.name}"
        out_lbl_path = labels_out / f"{args.name_prefix}{path.stem}.txt"
        rotated.save(out_img_path)
        out_lbl_path.write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
        )
        rotated_images += 1
        rotated_labels += 1
        render_progress(idx, len(image_paths))

    if image_paths:
        print()
    print(
        "Images rotated: "
        f"{rotated_images}, images skipped (no rotation): {kept_images}."
    )
    print(
        "Labels rotated: "
        f"{rotated_labels}, labels skipped (no rotation): {kept_labels}."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
