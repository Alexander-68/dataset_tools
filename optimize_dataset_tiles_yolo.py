#!/usr/bin/env python3
import argparse
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ASPECT_TOLERANCE = 0.01
SMALL_SQUARE_MAX = 224
MEDIUM_SQUARE_MAX = 384
DEFAULT_MAX_DIM = 1024
ROTATE_CHOICES = (0, 90, 270)
ROTATE_PAIR_CHOICES = (90, 270)
BACKGROUND_COLOR = (0, 0, 0)


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int

    @property
    def ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def max_dim(self) -> int:
        return max(self.width, self.height)


@dataclass(frozen=True)
class TileSpec:
    info: ImageInfo
    rotation: int


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize YOLO datasets by tiling images into mosaics and updating "
            "labels (detection or pose)."
        )
    )
    parser.add_argument("--images-dir", default="images", help="Input images folder.")
    parser.add_argument("--labels-dir", default="labels", help="Input labels folder.")
    parser.add_argument(
        "--images-out", default="images-opt", help="Output images folder."
    )
    parser.add_argument(
        "--labels-out", default="labels-opt", help="Output labels folder."
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help=(
            "If set, scale outputs so the max dimension is <= this value "
            f"(default: {DEFAULT_MAX_DIM}). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--out-ext",
        default=".jpg",
        help="Output image extension for mosaics (default: .jpg).",
    )
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


def format_eta(start_time: float, current: int, total: int) -> str:
    if total <= 0 or current <= 0:
        return "ETA --:--"
    elapsed = time.monotonic() - start_time
    if elapsed <= 0:
        return "ETA --:--"
    rate = current / elapsed
    if rate <= 0:
        return "ETA --:--"
    remaining = max(0.0, (total - current) / rate)
    eta_time = datetime.now() + timedelta(seconds=remaining)
    return f"ETA {eta_time.strftime('%H:%M')}"


def render_progress(current: int, total: int, label: str, start_time: float) -> None:
    if total <= 0:
        return
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(100 * current / total)
    tail = label
    if len(tail) > 40:
        tail = f"...{tail[-37:]}"
    eta_label = format_eta(start_time, current, total)
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} {percent:3d}% {eta_label} {tail}   "
    )
    sys.stdout.flush()


def ratio_matches(value: float, target: float, tol: float = ASPECT_TOLERANCE) -> bool:
    if target == 0:
        return False
    return abs(value - target) / target <= tol


def rotated_dims(width: int, height: int, angle: int) -> Tuple[int, int]:
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
    return x, y


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
) -> List[float]:
    out_parts: List[float] = []
    for idx in range(0, len(keypoints), 3):
        x, y, v = keypoints[idx], keypoints[idx + 1], keypoints[idx + 2]
        if v <= 0 or (x == 0.0 and y == 0.0):
            out_parts.extend([0.0, 0.0, 0.0])
            continue
        abs_x = x * width
        abs_y = y * height
        new_x, new_y = rotate_point(abs_x, abs_y, width, height, angle)
        if new_x < 0 or new_x > out_w or new_y < 0 or new_y > out_h:
            out_parts.extend([0.0, 0.0, 0.0])
            continue
        out_parts.extend(
            [
                clamp01(new_x / out_w),
                clamp01(new_y / out_h),
                v,
            ]
        )
    return out_parts


def rotate_labels(
    labels: Iterable[Tuple[str, List[float]]],
    width: int,
    height: int,
    angle: int,
) -> List[Tuple[str, List[float]]]:
    out_w, out_h = rotated_dims(width, height, angle)
    out_lines: List[Tuple[str, List[float]]] = []
    for class_id, values in labels:
        if len(values) < 4:
            continue
        if angle == 0:
            bbox = (
                clamp01(values[0]),
                clamp01(values[1]),
                clamp01(values[2]),
                clamp01(values[3]),
            )
        else:
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
        parts: List[float] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        if len(values) > 4:
            if angle == 0:
                kps = [clamp01(v) if (i % 3) != 2 else v for i, v in enumerate(values[4:])]
            else:
                kps = rotate_keypoints(values[4:], width, height, angle, out_w, out_h)
            parts.extend(kps)
        out_lines.append((class_id, parts))
    return out_lines


def scale_pad_labels(
    labels: Iterable[Tuple[str, List[float]]],
    src_w: int,
    src_h: int,
    tile_w: int,
    tile_h: int,
    scale_x: float,
    scale_y: float,
    pad_x: int,
    pad_y: int,
) -> List[Tuple[str, List[float]]]:
    out_lines: List[Tuple[str, List[float]]] = []
    for class_id, values in labels:
        if len(values) < 4:
            continue
        abs_cx = values[0] * src_w
        abs_cy = values[1] * src_h
        abs_w = values[2] * src_w
        abs_h = values[3] * src_h
        new_cx = abs_cx * scale_x + pad_x
        new_cy = abs_cy * scale_y + pad_y
        new_w = abs_w * scale_x
        new_h = abs_h * scale_y
        if new_w <= 0 or new_h <= 0:
            continue
        parts: List[float] = [
            clamp01(new_cx / tile_w),
            clamp01(new_cy / tile_h),
            clamp01(new_w / tile_w),
            clamp01(new_h / tile_h),
        ]
        if len(values) > 4:
            for idx in range(4, len(values), 3):
                x, y, v = values[idx], values[idx + 1], values[idx + 2]
                if v <= 0 or (x == 0.0 and y == 0.0):
                    parts.extend([0.0, 0.0, 0.0])
                    continue
                abs_x = x * src_w
                abs_y = y * src_h
                new_x = abs_x * scale_x + pad_x
                new_y = abs_y * scale_y + pad_y
                if new_x < 0 or new_x > tile_w or new_y < 0 or new_y > tile_h:
                    parts.extend([0.0, 0.0, 0.0])
                    continue
                parts.extend(
                    [
                        clamp01(new_x / tile_w),
                        clamp01(new_y / tile_h),
                        v,
                    ]
                )
        out_lines.append((class_id, parts))
    return out_lines


def offset_labels(
    labels: Iterable[Tuple[str, List[float]]],
    tile_w: int,
    tile_h: int,
    offset_x: int,
    offset_y: int,
    mosaic_w: int,
    mosaic_h: int,
) -> List[Tuple[str, List[float]]]:
    out_lines: List[Tuple[str, List[float]]] = []
    for class_id, values in labels:
        if len(values) < 4:
            continue
        abs_cx = values[0] * tile_w + offset_x
        abs_cy = values[1] * tile_h + offset_y
        abs_w = values[2] * tile_w
        abs_h = values[3] * tile_h
        parts: List[float] = [
            clamp01(abs_cx / mosaic_w),
            clamp01(abs_cy / mosaic_h),
            clamp01(abs_w / mosaic_w),
            clamp01(abs_h / mosaic_h),
        ]
        if len(values) > 4:
            for idx in range(4, len(values), 3):
                x, y, v = values[idx], values[idx + 1], values[idx + 2]
                if v <= 0 or (x == 0.0 and y == 0.0):
                    parts.extend([0.0, 0.0, 0.0])
                    continue
                abs_x = x * tile_w + offset_x
                abs_y = y * tile_h + offset_y
                parts.extend(
                    [
                        clamp01(abs_x / mosaic_w),
                        clamp01(abs_y / mosaic_h),
                        v,
                    ]
                )
        out_lines.append((class_id, parts))
    return out_lines


def labels_to_lines(labels: Iterable[Tuple[str, List[float]]]) -> List[str]:
    out_lines: List[str] = []
    for class_id, values in labels:
        if len(values) <= 4:
            parts = [class_id] + [format_float(v) for v in values[:4]]
        else:
            parts = [class_id] + [format_float(v) for v in values[:4]]
            for idx, value in enumerate(values[4:], start=4):
                if (idx - 4) % 3 == 2:
                    parts.append(format_vis(value))
                else:
                    parts.append(format_float(value))
        out_lines.append(" ".join(parts))
    return out_lines


def write_labels(label_path: Path, labels: Iterable[Tuple[str, List[float]]]) -> None:
    lines = labels_to_lines(labels)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def apply_rotation(img: Image.Image, angle: int) -> Image.Image:
    if angle == 90:
        return img.transpose(Image.ROTATE_270)
    if angle == 270:
        return img.transpose(Image.ROTATE_90)
    return img


def prepare_tile_image(
    path: Path, rotation: int, tile_w: int, tile_h: int
) -> Tuple[Image.Image, float, float, int, int]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = apply_rotation(img, rotation)
        rot_w, rot_h = img.size
        scale = min(tile_w / rot_w, tile_h / rot_h)
        new_w = max(1, int(round(rot_w * scale)))
        new_h = max(1, int(round(rot_h * scale)))
        scale_x = new_w / rot_w
        scale_y = new_h / rot_h
        pad_x = int((tile_w - new_w) / 2)
        pad_y = int((tile_h - new_h) / 2)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), color=BACKGROUND_COLOR)
        tile.paste(resized, (pad_x, pad_y))
    return tile, scale_x, scale_y, pad_x, pad_y


def maybe_resize_max_dim(
    image: Image.Image, max_dim: Optional[int]
) -> Image.Image:
    if max_dim is None or max_dim <= 0:
        return image
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    scale = max_dim / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return image.resize((new_w, new_h), Image.LANCZOS)


def build_groups(items: List[ImageInfo], group_size: int) -> List[List[ImageInfo]]:
    if not items:
        return []
    groups: List[List[ImageInfo]] = []
    idx = 0
    while idx < len(items):
        group = items[idx : idx + group_size]
        idx += group_size
        if len(group) < group_size:
            needed = group_size - len(group)
            fillers: List[ImageInfo] = []
            while len(fillers) < needed:
                for item in items:
                    fillers.append(item)
                    if len(fillers) == needed:
                        break
            group = group + fillers
        groups.append(group)
    return groups


def compute_tile_size(group: List[TileSpec]) -> Tuple[int, int]:
    widths: List[int] = []
    heights: List[int] = []
    for spec in group:
        w, h = spec.info.width, spec.info.height
        if spec.rotation in (90, 270):
            w, h = h, w
        widths.append(w)
        heights.append(h)
    return max(widths), max(heights)


def build_mosaic(
    group: List[TileSpec],
    cols: int,
    rows: int,
    labels_dir: Path,
    max_dim: Optional[int],
) -> Tuple[Image.Image, List[Tuple[str, List[float]]]]:
    tile_w, tile_h = compute_tile_size(group)
    mosaic_w = tile_w * cols
    mosaic_h = tile_h * rows
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), color=BACKGROUND_COLOR)
    out_labels: List[Tuple[str, List[float]]] = []

    for idx, spec in enumerate(group):
        offset_x = (idx % cols) * tile_w
        offset_y = (idx // cols) * tile_h
        tile, scale_x, scale_y, pad_x, pad_y = prepare_tile_image(
            spec.info.path, spec.rotation, tile_w, tile_h
        )
        mosaic.paste(tile, (offset_x, offset_y))

        labels = read_labels(labels_dir / f"{spec.info.path.stem}.txt")
        if labels:
            rot_w, rot_h = rotated_dims(
                spec.info.width, spec.info.height, spec.rotation
            )
            rotated = rotate_labels(
                labels, spec.info.width, spec.info.height, spec.rotation
            )
            scaled = scale_pad_labels(
                rotated, rot_w, rot_h, tile_w, tile_h, scale_x, scale_y, pad_x, pad_y
            )
            shifted = offset_labels(
                scaled, tile_w, tile_h, offset_x, offset_y, mosaic_w, mosaic_h
            )
            out_labels.extend(shifted)

    mosaic = maybe_resize_max_dim(mosaic, max_dim)
    return mosaic, out_labels


def rotate_mosaic(
    image: Image.Image,
    labels: Iterable[Tuple[str, List[float]]],
    angle: int,
) -> Tuple[Image.Image, List[Tuple[str, List[float]]]]:
    width, height = image.size
    rotated = apply_rotation(image, angle)
    out_labels = rotate_labels(labels, width, height, angle)
    return rotated, out_labels


def save_mosaic(
    image: Image.Image,
    labels: Iterable[Tuple[str, List[float]]],
    images_out: Path,
    labels_out: Path,
    name: str,
    out_ext: str,
) -> None:
    out_path = images_out / f"{name}{out_ext}"
    lbl_path = labels_out / f"{name}.txt"
    image.save(out_path)
    write_labels(lbl_path, labels)


def copy_or_resize_image(
    info: ImageInfo,
    images_out: Path,
    labels_dir: Path,
    labels_out: Path,
    max_dim: Optional[int],
) -> bool:
    src = info.path
    dst = images_out / src.name
    resized = False
    if max_dim is not None and max_dim > 0 and info.max_dim > max_dim:
        with Image.open(src) as img:
            width, height = img.size
            scale = max_dim / max(width, height)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            if img.mode in ("RGBA", "LA") and dst.suffix.lower() in {".jpg", ".jpeg"}:
                img = img.convert("RGB")
            img.resize((new_w, new_h), Image.LANCZOS).save(dst)
        resized = True
    else:
        shutil.copy2(src, dst)

    src_label = labels_dir / f"{src.stem}.txt"
    if src_label.exists():
        shutil.copy2(src_label, labels_out / src_label.name)
    return resized


def next_index(counters: Dict[str, int], prefix: str) -> int:
    counters[prefix] = counters.get(prefix, 0) + 1
    return counters[prefix]


def group_count(total: int, group_size: int) -> int:
    if total <= 0:
        return 0
    return (total + group_size - 1) // group_size


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)
    out_ext = args.out_ext if args.out_ext.startswith(".") else f".{args.out_ext}"
    max_dim = None if args.max_dim is None or args.max_dim <= 0 else args.max_dim

    print(
        "Optimizing dataset with tiled mosaics.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Images out: {images_out}\n"
        f"- Labels out: {labels_out}\n"
        f"- Max dim: {max_dim}\n"
        f"- Mosaic output ext: {out_ext}\n"
        f"- Aspect tolerance: {ASPECT_TOLERANCE * 100:.1f}%\n"
        f"- Square thresholds: <= {SMALL_SQUARE_MAX}px (t9), "
        f"<= {MEDIUM_SQUARE_MAX}px (t4)\n"
    )

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

    infos: List[ImageInfo] = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        infos.append(ImageInfo(path=path, width=width, height=height))

    small_squares: List[ImageInfo] = []
    medium_squares: List[ImageInfo] = []
    large_squares: List[ImageInfo] = []
    portrait_t2: List[ImageInfo] = []
    landscape_t2: List[ImageInfo] = []
    portrait_t6: List[ImageInfo] = []
    landscape_t6: List[ImageInfo] = []
    rest: List[ImageInfo] = []

    for info in infos:
        ratio = info.ratio
        if ratio_matches(ratio, 1.0):
            if info.max_dim <= SMALL_SQUARE_MAX:
                small_squares.append(info)
            elif info.max_dim <= MEDIUM_SQUARE_MAX:
                medium_squares.append(info)
            else:
                large_squares.append(info)
        elif ratio_matches(ratio, 0.5) or ratio_matches(ratio, 9 / 16):
            portrait_t2.append(info)
        elif ratio_matches(ratio, 2.0) or ratio_matches(ratio, 16 / 9):
            landscape_t2.append(info)
        elif ratio_matches(ratio, 2 / 3):
            portrait_t6.append(info)
        elif ratio_matches(ratio, 3 / 2):
            landscape_t6.append(info)
        else:
            rest.append(info)

    total_outputs = 0
    total_outputs += len(large_squares) + len(portrait_t6) + len(landscape_t6) + len(rest)
    total_outputs += group_count(len(small_squares), 9) * 2
    total_outputs += group_count(len(medium_squares), 4) * 2
    total_outputs += group_count(len(large_squares), 4) * 2
    total_outputs += group_count(len(portrait_t2), 2) * 2
    total_outputs += group_count(len(landscape_t2), 2) * 2
    total_outputs += group_count(len(portrait_t6), 6) * 2
    total_outputs += group_count(len(landscape_t6), 6) * 2
    total_outputs += group_count(len(rest), 4) * 2

    print(
        "Image buckets:\n"
        f"- Square <= {SMALL_SQUARE_MAX}px: {len(small_squares)}\n"
        f"- Square <= {MEDIUM_SQUARE_MAX}px: {len(medium_squares)}\n"
        f"- Square > {MEDIUM_SQUARE_MAX}px: {len(large_squares)}\n"
        f"- 1:2 / 9:16: {len(portrait_t2)}\n"
        f"- 2:1 / 16:9: {len(landscape_t2)}\n"
        f"- 2:3: {len(portrait_t6)}\n"
        f"- 3:2: {len(landscape_t6)}\n"
        f"- Other: {len(rest)}\n"
        f"- Planned outputs: {total_outputs}"
    )

    if total_outputs == 0:
        print("Nothing to do.")
        return 0

    start_time = time.monotonic()
    progress = 0
    counters: Dict[str, int] = {}
    stats: Dict[str, int] = {
        "as_is": 0,
        "as_is_resized": 0,
        "t9": 0,
        "t9r": 0,
        "t4": 0,
        "t4r": 0,
        "t2": 0,
        "t2r": 0,
        "t6": 0,
        "t6r": 0,
    }

    def bump(label: str) -> None:
        nonlocal progress
        progress += 1
        render_progress(progress, total_outputs, label, start_time)

    def mosaic_sets(
        items: List[ImageInfo],
        cols: int,
        rows: int,
        prefix: str,
        rotate_tiles: bool,
    ) -> int:
        if not items:
            return 0
        local = items[:]
        random.shuffle(local)
        groups = build_groups(local, cols * rows)
        count = 0
        for group in groups:
            specs: List[TileSpec] = []
            for info in group:
                rotation = random.choice(ROTATE_CHOICES) if rotate_tiles else 0
                specs.append(TileSpec(info=info, rotation=rotation))
            mosaic, labels = build_mosaic(specs, cols, rows, labels_dir, max_dim)
            idx = next_index(counters, prefix)
            name = f"{prefix}{idx:06d}"
            save_mosaic(mosaic, labels, images_out, labels_out, name, out_ext)
            bump(f"{name}{out_ext}")
            count += 1
        return count

    def mosaic_pairs(
        items: List[ImageInfo],
        cols: int,
        rows: int,
        prefix: str,
        rotate_prefix: str,
        angle_choices: Tuple[int, int],
    ) -> int:
        if not items:
            return 0
        local = items[:]
        random.shuffle(local)
        groups = build_groups(local, cols * rows)
        count = 0
        for group in groups:
            specs = [TileSpec(info=info, rotation=0) for info in group]
            mosaic, labels = build_mosaic(specs, cols, rows, labels_dir, max_dim)
            idx = next_index(counters, prefix)
            base_name = f"{prefix}{idx:06d}"
            save_mosaic(mosaic, labels, images_out, labels_out, base_name, out_ext)
            bump(f"{base_name}{out_ext}")
            count += 1

            angle = random.choice(angle_choices)
            rotated_img, rotated_labels = rotate_mosaic(mosaic, labels, angle)
            rot_name = f"{rotate_prefix}{idx:06d}"
            save_mosaic(rotated_img, rotated_labels, images_out, labels_out, rot_name, out_ext)
            bump(f"{rot_name}{out_ext}")
        return count

    stats["t9"] = mosaic_sets(small_squares, 3, 3, "t9_", rotate_tiles=False)
    stats["t9r"] = mosaic_sets(small_squares, 3, 3, "t9r_", rotate_tiles=True)

    stats["t4"] += mosaic_sets(medium_squares, 2, 2, "t4_", rotate_tiles=False)
    stats["t4r"] += mosaic_sets(medium_squares, 2, 2, "t4r_", rotate_tiles=True)

    for info in large_squares:
        if copy_or_resize_image(info, images_out, labels_dir, labels_out, max_dim):
            stats["as_is_resized"] += 1
        else:
            stats["as_is"] += 1
        bump(info.path.name)

    stats["t4"] += mosaic_sets(large_squares, 2, 2, "t4_", rotate_tiles=False)
    stats["t4r"] += mosaic_sets(large_squares, 2, 2, "t4r_", rotate_tiles=True)

    stats["t2"] += mosaic_pairs(
        portrait_t2, 2, 1, "t2_", "t2r_", ROTATE_PAIR_CHOICES
    )
    stats["t2"] += mosaic_pairs(
        landscape_t2, 1, 2, "t2_", "t2r_", ROTATE_PAIR_CHOICES
    )
    stats["t2r"] = stats["t2"]

    for info in portrait_t6 + landscape_t6:
        if copy_or_resize_image(info, images_out, labels_dir, labels_out, max_dim):
            stats["as_is_resized"] += 1
        else:
            stats["as_is"] += 1
        bump(info.path.name)

    stats["t6"] += mosaic_pairs(
        portrait_t6, 3, 2, "t6_", "t6r_", ROTATE_PAIR_CHOICES
    )
    stats["t6"] += mosaic_pairs(
        landscape_t6, 2, 3, "t6_", "t6r_", ROTATE_PAIR_CHOICES
    )
    stats["t6r"] = stats["t6"]

    for info in rest:
        if copy_or_resize_image(info, images_out, labels_dir, labels_out, max_dim):
            stats["as_is_resized"] += 1
        else:
            stats["as_is"] += 1
        bump(info.path.name)

    stats["t4"] += mosaic_sets(rest, 2, 2, "t4_", rotate_tiles=False)
    stats["t4r"] += mosaic_sets(rest, 2, 2, "t4r_", rotate_tiles=True)

    if total_outputs:
        print()

    print(
        "Done.\n"
        f"- Output images (as-is): {stats['as_is']}\n"
        f"- Output images (resized): {stats['as_is_resized']}\n"
        f"- t9: {stats['t9']}, t9r: {stats['t9r']}\n"
        f"- t4: {stats['t4']}, t4r: {stats['t4r']}\n"
        f"- t2: {stats['t2']}, t2r: {stats['t2r']}\n"
        f"- t6: {stats['t6']}, t6r: {stats['t6r']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
