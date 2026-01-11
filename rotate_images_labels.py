#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOWED_ANGLES = {90, 180, 270}


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def render_progress(current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ],
        key=lambda p: p.name.lower(),
    )


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
    return max(0.0, min(1.0, value))


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
    if len(parts) != expected:
        raise ValueError(
            f"Unexpected label length in {label_path} line {line_no}."
        )
    class_id = parts[0]
    try:
        values = [float(x) for x in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric label values in {label_path} line {line_no}."
        ) from exc
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


def parse_angles(raw: str) -> List[int]:
    angles: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            angle = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid angle: {part}") from exc
        if angle not in ALLOWED_ANGLES:
            raise ValueError(
                f"Angle must be one of {sorted(ALLOWED_ANGLES)}: {angle}"
            )
        if angle not in angles:
            angles.append(angle)
    if not angles:
        raise ValueError("No valid angles provided.")
    return angles


def rotate_dims(width: int, height: int, angle: int) -> Tuple[int, int]:
    if angle in (90, 270):
        return height, width
    return width, height


def rotate_point(
    x: float, y: float, width: float, height: float, angle: int
) -> Tuple[float, float]:
    if angle == 90:
        return height - y, x
    if angle == 180:
        return width - x, height - y
    if angle == 270:
        return y, width - x
    raise ValueError(f"Unsupported angle: {angle}")


def rotate_bbox(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
    angle: int,
    out_w: int,
    out_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    abs_cx = cx * width
    abs_cy = cy * height
    abs_w = bw * width
    abs_h = bh * height
    left = abs_cx - abs_w / 2.0
    right = abs_cx + abs_w / 2.0
    top = abs_cy - abs_h / 2.0
    bottom = abs_cy + abs_h / 2.0
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    rotated = [rotate_point(x, y, width, height, angle) for x, y in corners]
    xs = [pt[0] for pt in rotated]
    ys = [pt[1] for pt in rotated]
    min_x = max(0.0, min(xs))
    max_x = min(float(out_w), max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(float(out_h), max(ys))
    if max_x <= min_x or max_y <= min_y:
        return None
    new_cx = (min_x + max_x) / 2.0
    new_cy = (min_y + max_y) / 2.0
    new_w = max_x - min_x
    new_h = max_y - min_y
    return (
        clamp01(new_cx / out_w),
        clamp01(new_cy / out_h),
        clamp01(new_w / out_w),
        clamp01(new_h / out_h),
    )


def rotate_keypoints(
    keypoints: List[float],
    width: int,
    height: int,
    angle: int,
    out_w: int,
    out_h: int,
) -> List[str]:
    out_parts: List[str] = []
    for idx in range(0, len(keypoints), 3):
        x, y, v = keypoints[idx], keypoints[idx + 1], keypoints[idx + 2]
        if v <= 0 or (x == 0.0 and y == 0.0):
            out_parts.extend(["0", "0", "0"])
            continue
        abs_x = x * width
        abs_y = y * height
        new_x, new_y = rotate_point(abs_x, abs_y, width, height, angle)
        if new_x < 0 or new_x > out_w or new_y < 0 or new_y > out_h:
            out_parts.extend(["0", "0", "0"])
            continue
        out_parts.extend(
            [
                format_float(clamp01(new_x / out_w)),
                format_float(clamp01(new_y / out_h)),
                format_vis(v),
            ]
        )
    return out_parts


def transform_labels(
    labels: Iterable[Tuple[str, List[float]]],
    width: int,
    height: int,
    angle: int,
) -> List[str]:
    out_w, out_h = rotate_dims(width, height, angle)
    out_lines: List[str] = []
    for class_id, values in labels:
        if len(values) < 4:
            continue
        bbox = rotate_bbox(
            values[0],
            values[1],
            values[2],
            values[3],
            width,
            height,
            angle,
            out_w,
            out_h,
        )
        if bbox is None:
            continue
        parts = [
            class_id,
            format_float(bbox[0]),
            format_float(bbox[1]),
            format_float(bbox[2]),
            format_float(bbox[3]),
        ]
        if len(values) > 4:
            parts.extend(
                rotate_keypoints(values[4:], width, height, angle, out_w, out_h)
            )
        out_lines.append(" ".join(parts))
    return out_lines


def rotate_image(img: Image.Image, angle: int) -> Image.Image:
    if angle == 90:
        return img.transpose(Image.ROTATE_270)
    if angle == 180:
        return img.transpose(Image.ROTATE_180)
    if angle == 270:
        return img.transpose(Image.ROTATE_90)
    raise ValueError(f"Unsupported angle: {angle}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate images and YOLO labels by 90/180/270 degrees."
    )
    parser.add_argument("--images-dir", default="images", help="Input images.")
    parser.add_argument("--labels-dir", default="labels", help="Input labels.")
    parser.add_argument(
        "--images-out", default="images-rot", help="Output images folder."
    )
    parser.add_argument(
        "--labels-out", default="labels-rot", help="Output labels folder."
    )
    parser.add_argument(
        "--angles",
        default="90,180,270",
        help="Comma-separated clockwise angles: 90,180,270.",
    )
    parser.add_argument(
        "--prefix",
        default="r_",
        help="Prefix for output filenames (angle is inserted after prefix).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)
    angles = parse_angles(args.angles)

    print("Rotate images and labels (YOLO detection/pose).")
    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    print(f"Images: {images_dir} -> {images_out}")
    print(f"Labels: {labels_dir} -> {labels_out}")
    print(f"Angles (clockwise): {angles}")
    print(f"Output prefix: {args.prefix!r}")

    if not images_dir.exists():
        print(f"Error: Images folder not found: {images_dir}")
        return 1
    if not labels_dir.exists():
        print(f"Error: Labels folder not found: {labels_dir}")
        return 1

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(images_dir)
    if not image_paths:
        print("No images found.")
        return 0

    total_outputs = len(image_paths) * len(angles)
    processed_outputs = 0
    written_images = 0
    written_labels = 0
    images_failed = 0
    labels_missing = 0

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        labels = read_labels(label_path)
        if not label_path.exists():
            labels_missing += 1
        try:
            with Image.open(img_path) as im:
                base_img = im.convert("RGB")
            width, height = base_img.size
            for angle in angles:
                rotated_img = rotate_image(base_img, angle)
                out_lines = transform_labels(labels, width, height, angle)
                out_name = f"{args.prefix}{angle}_{img_path.name}"
                out_img_path = images_out / out_name
                rotated_img.save(out_img_path)
                written_images += 1
                if label_path.exists():
                    out_label_name = f"{args.prefix}{angle}_{img_path.stem}.txt"
                    out_lbl_path = labels_out / out_label_name
                    out_lbl_path.write_text(
                        "\n".join(out_lines) + ("\n" if out_lines else ""),
                        encoding="utf-8",
                    )
                    written_labels += 1
                processed_outputs += 1
                render_progress(processed_outputs, total_outputs)
        except Exception as exc:
            images_failed += 1
            print(f"\nError processing {img_path.name}: {exc}")
            processed_outputs += len(angles)
            render_progress(processed_outputs, total_outputs)

    if total_outputs:
        print()
    print(f"Images written: {written_images}. Failed images: {images_failed}.")
    print(
        "Labels written: "
        f"{written_labels}. Images missing labels: {labels_missing}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
