from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FACE_KPT_INDEXES = (0, 1, 2)  # nose, left eye, right eye
FACE_CENTER_KPT_INDEXES = (0, 1, 2, 3, 4)  # nose, eyes, ears
EAR_KPT_INDEXES = (3, 4)
MIN_PERSON_AREA_RATIO = 0.02
SECOND_PERSON_MIN_AREA_RATIO = 0.45


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def resolve_model_path(model_path: str, script_dir: Path) -> str:
    if "://" in model_path:
        return model_path
    candidate = Path(model_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(script_dir / candidate)


def list_images(source_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def render_progress(current: int, total: int, name: str) -> None:
    if total <= 0:
        return
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(100 * current / total)
    tail = name
    if len(tail) > 40:
        tail = f"...{tail[-37:]}"
    sys.stdout.write(f"\r[{bar}] {current}/{total} {percent:3d}% {tail}   ")
    sys.stdout.flush()


def count_person_detections(result) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return 0
    cls = getattr(boxes, "cls", None)
    if cls is None:
        return len(xyxy)
    try:
        return int((cls == 0).sum().item())
    except Exception:
        try:
            return sum(1 for c in cls.tolist() if int(c) == 0)
        except Exception:
            return len(xyxy)


def get_person_candidates(result) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    kpt_xy = getattr(keypoints, "xy", None)
    if xyxy is None or kpt_xy is None or len(kpt_xy) == 0:
        return []

    cls = getattr(boxes, "cls", None)
    conf = getattr(keypoints, "conf", None)
    candidates = []
    limit = min(len(xyxy), len(kpt_xy))
    for idx in range(limit):
        if cls is not None:
            try:
                if int(cls[idx]) != 0:
                    continue
            except Exception:
                try:
                    if int(cls[idx].item()) != 0:
                        continue
                except Exception:
                    continue
        person_xy = kpt_xy[idx]
        if len(person_xy) < max(FACE_CENTER_KPT_INDEXES) + 1:
            continue
        person_conf = conf[idx] if conf is not None else None
        if person_conf is not None:
            valid = True
            for kpt_idx in FACE_CENTER_KPT_INDEXES:
                if float(person_conf[kpt_idx]) <= 0:
                    valid = False
                    break
            if not valid:
                continue

        picked_points = []
        for kpt_idx in FACE_CENTER_KPT_INDEXES:
            x_val = float(person_xy[kpt_idx][0])
            y_val = float(person_xy[kpt_idx][1])
            picked_points.append((x_val, y_val))

        cx = sum(pt[0] for pt in picked_points) / len(picked_points)
        cy = sum(pt[1] for pt in picked_points) / len(picked_points)
        x1 = float(xyxy[idx][0])
        y1 = float(xyxy[idx][1])
        x2 = float(xyxy[idx][2])
        y2 = float(xyxy[idx][3])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        face_diameter = estimate_face_diameter(person_xy, person_conf)
        candidates.append(
            {
                "idx": idx,
                "area": area,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "face_diameter": face_diameter,
            }
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates


def keypoint_is_valid(conf: Optional[Iterable[float]], idx: int) -> bool:
    if conf is None:
        return True
    try:
        return float(conf[idx]) > 0
    except Exception:
        return False


def estimate_face_diameter(
    kpt_xy: list[list[float]],
    kpt_conf: Optional[Iterable[float]],
) -> Optional[float]:
    if len(kpt_xy) < 5:
        return None

    ear_points = []
    for ear_idx in EAR_KPT_INDEXES:
        if keypoint_is_valid(kpt_conf, ear_idx):
            ear_points.append((float(kpt_xy[ear_idx][0]), float(kpt_xy[ear_idx][1])))

    if len(ear_points) == 2:
        dx = ear_points[0][0] - ear_points[1][0]
        dy = ear_points[0][1] - ear_points[1][1]
        return math.hypot(dx, dy) * 1.32

    face_points = []
    for kp_idx in FACE_KPT_INDEXES:
        if kp_idx < len(kpt_xy) and keypoint_is_valid(kpt_conf, kp_idx):
            face_points.append((float(kpt_xy[kp_idx][0]), float(kpt_xy[kp_idx][1])))

    if len(face_points) < 2:
        return None

    max_dist = 0.0
    for i in range(len(face_points)):
        for j in range(i + 1, len(face_points)):
            dx = face_points[i][0] - face_points[j][0]
            dy = face_points[i][1] - face_points[j][1]
            max_dist = max(max_dist, math.hypot(dx, dy))
    return max_dist * 1.44 if max_dist > 0 else None


def square_crop_box(
    width: int,
    height: int,
    center: Tuple[float, float],
    side: Optional[int],
) -> Optional[Tuple[int, int, int, int]]:
    cx, cy = center
    if width <= 1 or height <= 1:
        return None
    cx = max(0.0, min(cx, width - 1))
    cy = max(0.0, min(cy, height - 1))
    if side is None:
        half = min(cx, cy, width - cx, height - cy)
        half_int = int(math.floor(half))
        if half_int < 1:
            return None
        side = 2 * half_int
    else:
        side = int(side)
        if side < 2:
            return None

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return (left, top, left + side, top + side)


def save_image(img: Image.Image, output_path: Path, size: Optional[int]) -> None:
    if size is not None and size > 0:
        img = img.resize((size, size), Image.LANCZOS)
    output_suffix = output_path.suffix.lower()
    save_img = img
    if output_suffix in {".jpg", ".jpeg"} and save_img.mode not in ("RGB", "L"):
        save_img = save_img.convert("RGB")
    save_kwargs = {}
    if output_suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"format": "JPEG", "quality": 95}
    save_img.save(output_path, **save_kwargs)


def crop_portraits(
    source_dir: Path,
    output_dir: Optional[Path],
    model_path: str,
    size: Optional[int],
    crop_percent: Optional[float],
    rotate: bool,
    debug_draw: bool,
    name_prefix: str,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    mode = "in-place" if output_dir is None else f"destination: {output_dir}"
    print(
        "Starting portrait crop.\n"
        f"- Source: {source_dir}\n"
        f"- Mode: {mode}\n"
        f"- Model: {model_path}\n"
        f"- Output size: {size}\n"
        f"- Crop percent: {crop_percent}\n"
        f"- Rotate: {rotate}\n"
        f"- Debug draw: {debug_draw}\n"
        f"- Name prefix: {name_prefix or '(none)'}"
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(source_dir)
    total_files = len(files)
    if total_files:
        print(f"Found {total_files} files. Processing...")
    else:
        print("No files found to process.")

    model = YOLO(model_path)
    processed = 0
    cropped = 0
    no_pose = 0
    skipped = 0

    for index, image_path in enumerate(files, start=1):
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if rotate:
                    angle = random.uniform(0.0, 45.0)
                    img = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
                width, height = img.size
                results = model.predict(img, verbose=False)
                result = results[0] if results else None
                person_count = count_person_detections(result)
                if person_count == 0:
                    no_pose += 1
                    processed += 1
                    render_progress(index, total_files, image_path.name)
                    continue
                candidates = get_person_candidates(result)
                if not candidates:
                    no_pose += 1
                    processed += 1
                    render_progress(index, total_files, image_path.name)
                    continue
                primary = candidates[0]
                image_area = width * height
                min_area = image_area * MIN_PERSON_AREA_RATIO
                large_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["area"] >= min_area
                    and candidate["face_diameter"] is not None
                ]
                secondary = None
                if (
                    len(large_candidates) >= 2
                    and large_candidates[1]["area"]
                    >= large_candidates[0]["area"] * SECOND_PERSON_MIN_AREA_RATIO
                ):
                    secondary = large_candidates[1]

                crop_box = None
                if secondary is not None:
                    primary_center = primary["center"]
                    secondary_center = secondary["center"]
                    primary_radius = primary["face_diameter"] / 2.0
                    secondary_radius = secondary["face_diameter"] / 2.0
                    min_x = min(
                        primary_center[0] - primary_radius,
                        secondary_center[0] - secondary_radius,
                    )
                    min_y = min(
                        primary_center[1] - primary_radius,
                        secondary_center[1] - secondary_radius,
                    )
                    max_x = max(
                        primary_center[0] + primary_radius,
                        secondary_center[0] + secondary_radius,
                    )
                    max_y = max(
                        primary_center[1] + primary_radius,
                        secondary_center[1] + secondary_radius,
                    )
                    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
                    crop_side = max(max_x - min_x, max_y - min_y)
                    if crop_percent is not None:
                        crop_side *= 1.0 + crop_percent / 100.0
                    crop_side = int(round(crop_side))
                    max_side = min(width, height)
                    if crop_side > max_side:
                        crop_side = max_side
                    crop_box = square_crop_box(width, height, center, crop_side)
                else:
                    center = primary["center"]
                    face_diameter = primary["face_diameter"]
                    crop_side = None
                    if crop_percent is not None and face_diameter is not None:
                        crop_side = face_diameter * (1.0 + crop_percent / 100.0)
                        crop_side = int(round(crop_side))
                        max_side = min(width, height)
                        if crop_side > max_side:
                            crop_side = max_side
                    crop_box = square_crop_box(width, height, center, crop_side)

                if crop_box is None:
                    skipped += 1
                    processed += 1
                    render_progress(index, total_files, image_path.name)
                    continue
                if debug_draw:
                    draw = ImageDraw.Draw(img)
                    active = [primary, secondary] if secondary is not None else [primary]
                    for candidate in active:
                        if candidate is None:
                            continue
                        face_diameter = candidate["face_diameter"]
                        if face_diameter is None:
                            continue
                        center = candidate["center"]
                        radius = face_diameter / 2.0
                        left = center[0] - radius
                        top = center[1] - radius
                        right = center[0] + radius
                        bottom = center[1] + radius
                        draw.ellipse(
                            [left, top, right, bottom],
                            outline="white",
                            width=3,
                        )

                cropped_img = img.crop(crop_box)

                output_name = (
                    f"{name_prefix}{image_path.name}" if name_prefix else image_path.name
                )
                target_dir = image_path.parent if output_dir is None else output_dir
                output_path = target_dir / output_name
                save_image(cropped_img, output_path, size)
                cropped += 1
                processed += 1
        except (UnidentifiedImageError, OSError):
            skipped += 1
        finally:
            render_progress(index, total_files, image_path.name)

    if total_files:
        print()

    message = f"Scanned {total_files} files. Processed {processed} images."
    if cropped:
        message += f" Cropped {cropped} images."
    if no_pose:
        message += f" No pose detected in {no_pose} images."
    if skipped:
        message += f" Skipped {skipped} files."
    if output_dir:
        message += f" Saved to {output_dir}."
    print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Square-crop portrait images by centering on face keypoints "
            "(nose + eyes) from a YOLO pose model."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("images"),
        help="Input images folder. Default: ./images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output folder. If omitted, crops in-place.",
    )
    parser.add_argument(
        "--model",
        default="yolo11x-pose.pt",
        help="YOLO pose model path or name. Default: yolo11x-pose.pt",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Optional output size for the square crop in pixels.",
    )
    parser.add_argument(
        "--crop-percent",
        type=float,
        default=None,
        help=(
            "Optional percent to expand crop size from face diameter "
            "(e.g., 20 means fd * 1.2). Default: max possible square."
        ),
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Random clockwise rotation between 10 and 45 degrees before cropping.",
    )
    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help="Draw a white face circle (center + diameter) before cropping.",
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Optional prefix for output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    input_dir = resolve_path(args.input, cwd)
    output_dir = resolve_path(args.output, cwd) if args.output else None
    model_path = resolve_model_path(args.model, script_dir)
    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    crop_portraits(
        input_dir,
        output_dir,
        model_path,
        args.size,
        args.crop_percent,
        args.rotate,
        args.debug_draw,
        args.name_prefix,
    )


if __name__ == "__main__":
    main()
