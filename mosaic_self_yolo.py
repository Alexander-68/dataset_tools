#!/usr/bin/env python3
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageOps

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
DEFAULT_MOSAIC_FACTOR = 0.7
DEFAULT_MIN_BBOX_PIXELS = 20


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


@dataclass(frozen=True)
class TilePlacement:
    offset_x: int
    offset_y: int
    tile_w: int
    tile_h: int
    rotation: int
    flip: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build self-mosaics with a short-axis extension factor and "
            "transform YOLO labels (detection or pose with 17 keypoints)."
        )
    )
    parser.add_argument("--images-dir", default="images", help="Input images folder.")
    parser.add_argument("--labels-dir", default="labels", help="Input labels folder.")
    parser.add_argument(
        "--images-out", default="images_mosaic", help="Output images folder."
    )
    parser.add_argument(
        "--labels-out", default="labels_mosaic", help="Output labels folder."
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="If set, scale mosaics so the max dimension equals this value.",
    )
    parser.add_argument(
        "--mosaic-scale",
        type=float,
        default=DEFAULT_MOSAIC_FACTOR,
        help="Short-axis extension factor applied to the long axis (default: 0.7).",
    )
    parser.add_argument(
        "--min-bbox-pixels",
        type=int,
        default=DEFAULT_MIN_BBOX_PIXELS,
        help=(
            "Drop boxes smaller than this size in pixels (default: 20). "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--out-ext",
        default=".jpg",
        help="Output image extension (default: .jpg).",
    )
    parser.add_argument(
        "--prefix",
        default="s_",
        help="Output filename prefix (default: s_).",
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
) -> Tuple[str, List[float]]:
    parts = line.strip().split()
    if not parts:
        return None  # type: ignore[return-value]
    if len(parts) == 5:
        expected = 5
    elif len(parts) == 56:
        expected = 56
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


def apply_transform(img: Image.Image, rotation: int, flip: bool) -> Image.Image:
    out = img
    if flip:
        out = ImageOps.mirror(out)
    if rotation == 90:
        out = out.transpose(Image.ROTATE_90)
    elif rotation == 270:
        out = out.transpose(Image.ROTATE_270)
    return out


def rotated_size(width: int, height: int, rotation: int) -> Tuple[int, int]:
    if rotation in (90, 270):
        return height, width
    return width, height


def apply_transform_fit(
    img: Image.Image, rotation: int, flip: bool, target_w: int, target_h: int
) -> Image.Image:
    oriented = apply_transform(img, rotation, flip)
    return oriented.resize((target_w, target_h), Image.LANCZOS)


def mosaic_extra_dim(long_dim: int, factor: float) -> int:
    extra = int(long_dim * factor)
    return max(1, extra)


def fit_tile_size(
    rot_w: int, rot_h: int, rem_w: int, rem_h: int, prefer: str
) -> Tuple[int, int, str]:
    if rem_w <= 0 or rem_h <= 0:
        raise ValueError("Remaining space too small to place mosaic tiles.")
    if prefer == "height":
        tile_h = rem_h
        tile_w = int(round(rot_w * rem_h / rot_h))
        filled = "height"
        if tile_w > rem_w:
            tile_w = rem_w
            tile_h = int(round(rot_h * rem_w / rot_w))
            filled = "width"
    else:
        tile_w = rem_w
        tile_h = int(round(rot_h * rem_w / rot_w))
        filled = "width"
        if tile_h > rem_h:
            tile_h = rem_h
            tile_w = int(round(rot_w * rem_h / rot_h))
            filled = "height"
    tile_w = max(1, min(tile_w, rem_w))
    tile_h = max(1, min(tile_h, rem_h))
    return tile_w, tile_h, filled


def build_mosaic_tiles(
    width: int, height: int, portrait: bool, factor: float
) -> Tuple[List[TilePlacement], Tuple[int, int]]:
    extra = mosaic_extra_dim(height if portrait else width, factor)
    placements: List[TilePlacement] = []
    if portrait:
        mosaic_w = width + extra
        mosaic_h = height
        rem_x, rem_y = width, 0
        rem_w, rem_h = extra, height
        steps = [
            {"rotation": 90, "flip": False, "prefer": "width"},
            {"rotation": 0, "flip": True, "prefer": "height"},
            {"rotation": 270, "flip": False, "prefer": "width"},
            {"rotation": 0, "flip": False, "prefer": "height"},
            {"rotation": 0, "flip": True, "prefer": "height"},
        ]
    else:
        mosaic_w = width
        mosaic_h = height + extra
        rem_x, rem_y = 0, height
        rem_w, rem_h = width, extra
        steps = [
            {"rotation": 90, "flip": False, "prefer": "height"},
            {"rotation": 0, "flip": True, "prefer": "width"},
            {"rotation": 270, "flip": False, "prefer": "height"},
            {"rotation": 0, "flip": False, "prefer": "width"},
            {"rotation": 0, "flip": True, "prefer": "width"},
        ]

    for step_idx, step in enumerate(steps, start=1):
        if rem_w <= 0 or rem_h <= 0:
            break
        rot_w, rot_h = rotated_size(width, height, step["rotation"])
        tile_w, tile_h, filled = fit_tile_size(
            rot_w, rot_h, rem_w, rem_h, step["prefer"]
        )
        placements.append(
            TilePlacement(
                offset_x=rem_x,
                offset_y=rem_y,
                tile_w=tile_w,
                tile_h=tile_h,
                rotation=step["rotation"],
                flip=step["flip"],
            )
        )
        if filled == "height":
            rem_x += tile_w
            rem_w -= tile_w
        else:
            rem_y += tile_h
            rem_h -= tile_h
    return placements, (mosaic_w, mosaic_h)


def transform_point_abs(
    x_abs: float,
    y_abs: float,
    placement: TilePlacement,
    orig_w: int,
    orig_h: int,
    scale_x: float,
    scale_y: float,
) -> Tuple[float, float]:
    if placement.flip:
        x_abs = orig_w - x_abs
    if placement.rotation == 90:
        x_rot = y_abs
        y_rot = orig_w - x_abs
    elif placement.rotation == 270:
        x_rot = orig_h - y_abs
        y_rot = x_abs
    else:
        x_rot = x_abs
        y_rot = y_abs
    x_scaled = x_rot * scale_x
    y_scaled = y_rot * scale_y
    x_mosaic = placement.offset_x + x_scaled
    y_mosaic = placement.offset_y + y_scaled
    return x_mosaic, y_mosaic


def transform_bbox(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    placement: TilePlacement,
    orig_w: int,
    orig_h: int,
    scale_mosaic_x: float,
    scale_mosaic_y: float,
    final_w: int,
    final_h: int,
    scale_x: float,
    scale_y: float,
    min_bbox_pixels: int,
) -> Optional[Tuple[float, float, float, float]]:
    cx_abs = cx * orig_w
    cy_abs = cy * orig_h
    bw_abs = bw * orig_w
    bh_abs = bh * orig_h
    x1 = cx_abs - bw_abs / 2.0
    y1 = cy_abs - bh_abs / 2.0
    x2 = cx_abs + bw_abs / 2.0
    y2 = cy_abs + bh_abs / 2.0
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    xs: List[float] = []
    ys: List[float] = []
    for x_abs, y_abs in corners:
        x_mosaic, y_mosaic = transform_point_abs(
            x_abs,
            y_abs,
            placement,
            orig_w,
            orig_h,
            scale_x,
            scale_y,
        )
        xs.append(x_mosaic * scale_mosaic_x)
        ys.append(y_mosaic * scale_mosaic_y)
    min_x = max(0.0, min(xs))
    max_x = min(float(final_w), max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(float(final_h), max(ys))
    if max_x <= min_x or max_y <= min_y:
        return None
    out_cx = (min_x + max_x) / 2.0
    out_cy = (min_y + max_y) / 2.0
    out_w = max_x - min_x
    out_h = max_y - min_y
    if min_bbox_pixels > 0 and (out_w < min_bbox_pixels or out_h < min_bbox_pixels):
        return None
    return (
        clamp01(out_cx / final_w),
        clamp01(out_cy / final_h),
        clamp01(out_w / final_w),
        clamp01(out_h / final_h),
    )


def transform_keypoints(
    keypoints: List[Tuple[float, float, float]],
    placement: TilePlacement,
    orig_w: int,
    orig_h: int,
    scale_mosaic_x: float,
    scale_mosaic_y: float,
    final_w: int,
    final_h: int,
    scale_x: float,
    scale_y: float,
) -> List[str]:
    if placement.flip:
        keypoints = [keypoints[i] for i in FLIP_IDX]
    out: List[str] = []
    for x, y, v in keypoints:
        if x == 0.0 and y == 0.0 and v == 0.0:
            out.extend(["0", "0", "0"])
            continue
        x_abs = x * orig_w
        y_abs = y * orig_h
        x_mosaic, y_mosaic = transform_point_abs(
            x_abs,
            y_abs,
            placement,
            orig_w,
            orig_h,
            scale_x,
            scale_y,
        )
        x_final = x_mosaic * scale_mosaic_x
        y_final = y_mosaic * scale_mosaic_y
        out.extend(
            [
                format_float(clamp01(x_final / final_w)),
                format_float(clamp01(y_final / final_h)),
                format_vis(v),
            ]
        )
    return out


def transform_labels(
    labels: List[Tuple[str, List[float]]],
    placements: List[TilePlacement],
    orig_size: Tuple[int, int],
    scale_mosaic_x: float,
    scale_mosaic_y: float,
    final_size: Tuple[int, int],
    min_bbox_pixels: int,
) -> Tuple[List[str], int]:
    orig_w, orig_h = orig_size
    final_w, final_h = final_size
    out_lines: List[str] = []
    used_count = 0

    for placement in placements:
        rot_w, rot_h = rotated_size(orig_w, orig_h, placement.rotation)
        scale_x = placement.tile_w / rot_w
        scale_y = placement.tile_h / rot_h
        placement_lines: List[str] = []
        for class_id, values in labels:
            bbox = transform_bbox(
                values[0],
                values[1],
                values[2],
                values[3],
                placement,
                orig_w,
                orig_h,
                scale_mosaic_x,
                scale_mosaic_y,
                final_w,
                final_h,
                scale_x,
                scale_y,
                min_bbox_pixels,
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
                kps = values[4:]
                if len(kps) != 51:
                    raise ValueError(
                        f"Expected 17 keypoints (51 values), got {len(kps)}."
                    )
                keypoints = [
                    (kps[idx], kps[idx + 1], kps[idx + 2])
                    for idx in range(0, len(kps), 3)
                ]
                parts.extend(
                    transform_keypoints(
                        keypoints,
                        placement,
                        orig_w,
                        orig_h,
                        scale_mosaic_x,
                        scale_mosaic_y,
                        final_w,
                        final_h,
                        scale_x,
                        scale_y,
                    )
                )
            placement_lines.append(" ".join(parts))
        if not placement_lines:
            break
        out_lines.extend(placement_lines)
        used_count += 1
    return out_lines, used_count


def is_portrait(width: int, height: int) -> bool:
    return height >= width


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)
    out_ext = args.out_ext if args.out_ext.startswith(".") else f".{args.out_ext}"
    prefix = args.prefix or ""
    if args.mosaic_scale <= 0:
        raise ValueError("--mosaic-scale must be a positive number.")
    if args.min_bbox_pixels < 0:
        raise ValueError("--min-bbox-pixels must be zero or a positive integer.")

    print(
        "Building self-mosaics from each image.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Images out: {images_out}\n"
        f"- Labels out: {labels_out}\n"
        f"- Max dim: {args.max_dim}\n"
        f"- Mosaic scale: {args.mosaic_scale}\n"
        f"- Min bbox pixels: {args.min_bbox_pixels}"
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

    total_images = len(image_paths)
    bar_width = 30
    for idx, img_path in enumerate(image_paths, start=1):
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        orig_w, orig_h = img.size
        portrait = is_portrait(orig_w, orig_h)
        placements, mosaic_size = build_mosaic_tiles(
            orig_w, orig_h, portrait, args.mosaic_scale
        )
        mosaic_w, mosaic_h = mosaic_size
        final_w, final_h = mosaic_w, mosaic_h
        scale_mosaic_x = 1.0
        scale_mosaic_y = 1.0
        if args.max_dim is not None:
            if args.max_dim <= 0:
                raise ValueError("--max-dim must be a positive integer.")
            scale = args.max_dim / max(mosaic_w, mosaic_h)
            final_w = max(1, int(round(mosaic_w * scale)))
            final_h = max(1, int(round(mosaic_h * scale)))
            scale_mosaic_x = final_w / mosaic_w
            scale_mosaic_y = final_h / mosaic_h

        labels = read_labels(labels_dir / f"{img_path.stem}.txt")
        placements_with_original = [
            TilePlacement(0, 0, orig_w, orig_h, 0, False),
            *placements,
        ]
        out_lines, used_count = transform_labels(
            labels,
            placements_with_original,
            (orig_w, orig_h),
            scale_mosaic_x,
            scale_mosaic_y,
            (final_w, final_h),
            args.min_bbox_pixels,
        )

        if not out_lines:
            filled = int(round(bar_width * idx / total_images))
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(f"\r[{bar}] {idx}/{total_images}")
            sys.stdout.flush()
            continue

        mosaic = Image.new("RGB", mosaic_size, (0, 0, 0))
        mosaic.paste(img, (0, 0))

        for placement in placements[: max(0, used_count - 1)]:
            tile = apply_transform_fit(
                img,
                placement.rotation,
                placement.flip,
                placement.tile_w,
                placement.tile_h,
            )
            mosaic.paste(tile, (placement.offset_x, placement.offset_y))

        if scale_mosaic_x != 1.0 or scale_mosaic_y != 1.0:
            mosaic = mosaic.resize((final_w, final_h), Image.LANCZOS)

        out_img_path = images_out / f"{prefix}{img_path.stem}{out_ext}"
        out_lbl_path = labels_out / f"{prefix}{img_path.stem}.txt"

        mosaic.save(out_img_path)
        out_lbl_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

        filled = int(round(bar_width * idx / total_images))
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r[{bar}] {idx}/{total_images}")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
