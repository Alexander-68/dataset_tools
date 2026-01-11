#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
MIN_BBOX_PIXELS = 20


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build mosaics from image groups and merge YOLO labels "
            "(detection or pose with 17 keypoints)."
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
        "--out-ext",
        default=".jpg",
        help="Output image extension (default: .jpg).",
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Optional prefix to add to generated file names.",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip each source image left-right before building the mosaic.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help=(
            "Randomly rotate each source image by 90 or 270 degrees "
            "before building the mosaic."
        ),
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


def parse_label_line(line: str, label_path: Path, line_no: int) -> Tuple[str, List[float]]:
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


def is_square(width: int, height: int) -> bool:
    min_dim = min(width, height)
    max_dim = max(width, height)
    if max_dim == 0:
        return False
    return (min_dim / max_dim) > 0.8


def is_portrait(width: int, height: int) -> bool:
    return height > width


def partition_by_shape(
    image_paths: List[Path],
) -> Tuple[List[Path], List[Path], List[Path]]:
    squares: List[Path] = []
    portraits: List[Path] = []
    landscapes: List[Path] = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        if is_square(width, height):
            squares.append(path)
        elif is_portrait(width, height):
            portraits.append(path)
        else:
            landscapes.append(path)
    return squares, portraits, landscapes


def build_grid_mosaic(
    images: List[Image.Image], cols: int, rows: int
) -> Tuple[Image.Image, List[float], List[Tuple[int, int]], Tuple[int, int]]:
    if len(images) != cols * rows:
        raise ValueError(
            f"Expected {cols * rows} images, got {len(images)}."
        )

    orig_sizes = [img.size for img in images]
    row_images: List[List[Image.Image]] = []
    row_widths: List[int] = []
    for r in range(rows):
        start = r * cols
        target_h = images[start].height
        row: List[Image.Image] = []
        for c in range(cols):
            img = images[start + c]
            if img.height != target_h:
                scale = target_h / img.height
                new_w = max(1, int(round(img.width * scale)))
                img = img.resize((new_w, target_h), Image.LANCZOS)
            row.append(img)
        row_images.append(row)
        row_widths.append(sum(im.width for im in row))

    base_row_w = row_widths[0]
    scaled_rows: List[List[Image.Image]] = []
    for r, row in enumerate(row_images):
        row_scale = 1.0
        if row_widths[r] != base_row_w:
            row_scale = base_row_w / row_widths[r]
        scaled_row: List[Image.Image] = []
        for img in row:
            if row_scale != 1.0:
                new_w = max(1, int(round(img.width * row_scale)))
                new_h = max(1, int(round(img.height * row_scale)))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            scaled_row.append(img)
        scaled_rows.append(scaled_row)

    offsets: List[Tuple[int, int]] = []
    scales: List[float] = []
    mosaic_w = 0
    y = 0
    idx = 0
    for r in range(rows):
        x = 0
        row_h = scaled_rows[r][0].height
        row_w = sum(im.width for im in scaled_rows[r])
        mosaic_w = max(mosaic_w, row_w)
        for c in range(cols):
            img = scaled_rows[r][c]
            offsets.append((x, y))
            orig_w, orig_h = orig_sizes[idx]
            scale = img.width / orig_w if orig_w else 1.0
            scales.append(scale)
            x += img.width
            idx += 1
        y += row_h

    mosaic_h = y
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h))
    y = 0
    for r in range(rows):
        x = 0
        for c in range(cols):
            img = scaled_rows[r][c]
            mosaic.paste(img, (x, y))
            x += img.width
        y += scaled_rows[r][0].height

    return mosaic, scales, offsets, (mosaic_w, mosaic_h)


def transform_labels(
    labels: List[Tuple[str, List[float]]],
    orig_size: Tuple[int, int],
    rotated_size: Tuple[int, int],
    scale_img: float,
    offset: Tuple[int, int],
    scale_mosaic_x: float,
    scale_mosaic_y: float,
    mosaic_size: Tuple[int, int],
    flip: bool,
    rotate_angle: int,
) -> List[str]:
    orig_w, orig_h = orig_size
    rot_w, rot_h = rotated_size
    offset_x, offset_y = offset
    mosaic_w, mosaic_h = mosaic_size
    out_lines: List[str] = []
    if rotate_angle not in (0, 90, 270):
        raise ValueError(f"Unsupported rotate angle: {rotate_angle}")

    def rotate_point(x: float, y: float) -> Tuple[float, float]:
        if rotate_angle == 90:
            return y, orig_w - x
        if rotate_angle == 270:
            return orig_h - y, x
        return x, y

    for class_id, values in labels:
        cx, cy, w, h = values[:4]
        abs_cx = cx * orig_w
        abs_cy = cy * orig_h
        abs_w = w * orig_w
        abs_h = h * orig_h
        if rotate_angle:
            abs_cx, abs_cy = rotate_point(abs_cx, abs_cy)
            abs_w, abs_h = abs_h, abs_w
        cx = abs_cx / rot_w if rot_w else 0.0
        cy = abs_cy / rot_h if rot_h else 0.0
        w = abs_w / rot_w if rot_w else 0.0
        h = abs_h / rot_h if rot_h else 0.0
        if flip:
            cx = 1.0 - cx
        abs_cx = (cx * rot_w * scale_img + offset_x) * scale_mosaic_x
        abs_cy = (cy * rot_h * scale_img + offset_y) * scale_mosaic_y
        abs_w = (w * rot_w * scale_img) * scale_mosaic_x
        abs_h = (h * rot_h * scale_img) * scale_mosaic_y
        if abs_w < MIN_BBOX_PIXELS or abs_h < MIN_BBOX_PIXELS:
            continue
        norm_cx = clamp01(abs_cx / mosaic_w)
        norm_cy = clamp01(abs_cy / mosaic_h)
        norm_w = clamp01(abs_w / mosaic_w)
        norm_h = clamp01(abs_h / mosaic_h)
        parts = [
            class_id,
            format_float(norm_cx),
            format_float(norm_cy),
            format_float(norm_w),
            format_float(norm_h),
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
            rotated_kps: List[Tuple[float, float, float]] = []
            for x, y, v in keypoints:
                if x == 0.0 and y == 0.0 and v == 0.0:
                    rotated_kps.append((0.0, 0.0, v))
                    continue
                abs_x = x * orig_w
                abs_y = y * orig_h
                if rotate_angle:
                    abs_x, abs_y = rotate_point(abs_x, abs_y)
                x = abs_x / rot_w if rot_w else 0.0
                y = abs_y / rot_h if rot_h else 0.0
                rotated_kps.append((x, y, v))
            if flip:
                rotated_kps = [rotated_kps[i] for i in FLIP_IDX]
            for x, y, v in rotated_kps:
                if x == 0.0 and y == 0.0 and v == 0.0:
                    parts.extend(["0", "0", "0"])
                else:
                    if flip:
                        x = 1.0 - x
                    abs_x = (x * rot_w * scale_img + offset_x) * scale_mosaic_x
                    abs_y = (y * rot_h * scale_img + offset_y) * scale_mosaic_y
                    norm_x = clamp01(abs_x / mosaic_w)
                    norm_y = clamp01(abs_y / mosaic_h)
                    parts.extend(
                        [format_float(norm_x), format_float(norm_y), format_vis(v)]
                    )
        out_lines.append(" ".join(parts))
    return out_lines


def render_progress(current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)
    out_ext = args.out_ext if args.out_ext.startswith(".") else f".{args.out_ext}"

    print(
        "Building mosaics from image groups.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Images out: {images_out}\n"
        f"- Labels out: {labels_out}\n"
        f"- Max dim: {args.max_dim}\n"
        f"- Flip: {args.flip}, Rotate: {args.rotate}"
    )

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(images_dir)
    random.shuffle(image_paths)
    if len(image_paths) < 2:
        print("Need at least two images to build mosaics.")
        return 1
    image_index = {path: idx for idx, path in enumerate(image_paths, start=1)}

    groups: List[Tuple[Tuple[Path, ...], int, int]] = []

    def add_groups(
        paths: List[Path], group_size: int, label: str, cols: int, rows: int
    ) -> None:
        if not paths:
            return
        local_groups = [
            tuple(paths[i : i + group_size])
            for i in range(0, len(paths) - (group_size - 1), group_size)        
        ]
        remainder = len(paths) % group_size
        if remainder:
            leftovers = paths[-remainder:]
            if local_groups:
                used_pool = [p for group in local_groups for p in group]
                fillers = [random.choice(used_pool) for _ in range(group_size - remainder)]
                completed = tuple(leftovers + fillers)
                print(
                    f"Completing unpaired {label} images: "
                    f"{', '.join(p.name for p in leftovers)}"
                )
                local_groups.append(completed)
            else:
                print(
                    f"Skipping unpaired {label} images: "
                    f"{', '.join(p.name for p in leftovers)}"
                )
        if not local_groups:
            return
        groups.extend((group, cols, rows) for group in local_groups)

    if args.rotate:
        add_groups(image_paths, 4, "mixed", 2, 2)
    else:
        square_paths, portrait_paths, landscape_paths = partition_by_shape(
            image_paths
        )
        add_groups(square_paths, 4, "square", 2, 2)
        add_groups(portrait_paths, 6, "vertical", 3, 2)
        add_groups(landscape_paths, 6, "horizontal", 2, 3)

    if not groups:
        print("Need at least one full group of images to build mosaics.")
        return 1

    total_groups = len(groups)
    for idx, group_entry in enumerate(groups, start=1):
        group, cols, rows = group_entry
        images: List[Image.Image] = []
        orig_sizes: List[Tuple[int, int]] = []
        rotated_sizes: List[Tuple[int, int]] = []
        rotation_angles: List[int] = []
        for path in group:
            with Image.open(path) as im:
                im = im.convert("RGB")
                orig_sizes.append(im.size)
                angle = 0
                if args.rotate:
                    roll = random.random()
                    if roll < 0.25:
                        angle = 90
                    elif roll < 0.5:
                        angle = 270
                if angle:
                    im = im.rotate(angle, expand=True)
                if args.flip:
                    im = ImageOps.mirror(im)
                images.append(im)
                rotation_angles.append(angle)
                rotated_sizes.append(im.size)

        mosaic, scales, offsets, mosaic_size = build_grid_mosaic(
            images, cols, rows
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
            if scale_mosaic_x != 1.0 or scale_mosaic_y != 1.0:
                mosaic = mosaic.resize((final_w, final_h), Image.LANCZOS)

        out_lines: List[str] = []
        for img, path, scale, offset, orig_size, rotated_size, angle in zip(
            images, group, scales, offsets, orig_sizes, rotated_sizes, rotation_angles
        ):
            labels = read_labels(labels_dir / f"{path.stem}.txt")
            out_lines.extend(
                transform_labels(
                    labels,
                    orig_size,
                    rotated_size,
                    scale,
                    offset,
                    scale_mosaic_x,
                    scale_mosaic_y,
                    (final_w, final_h),
                    args.flip,
                    angle,
                )
            )

        out_stem = "_".join(str(image_index[p]) for p in group)
        if args.name_prefix:
            out_stem = f"{args.name_prefix}{out_stem}"
        out_img_path = images_out / f"{out_stem}{out_ext}"
        out_lbl_path = labels_out / f"{out_stem}.txt"

        mosaic.save(out_img_path)
        out_lbl_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        render_progress(idx, total_groups)

    if total_groups:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
