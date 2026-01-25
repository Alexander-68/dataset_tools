from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency
    HEIF_SUPPORTED = False
else:
    register_heif_opener()
    HEIF_SUPPORTED = True


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def format_eta(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return "ETA --:--"
    total_seconds = int(round(seconds))
    if total_seconds >= 3600:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"ETA {hours:02d}:{minutes:02d}:{secs:02d}"
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"ETA {minutes:02d}:{secs:02d}"


def render_progress(
    current: int, total: int, start_time: float, label: str
) -> None:
    if total <= 0:
        return
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(100 * current / total)
    elapsed = time.time() - start_time
    eta = None
    if current > 0:
        eta = (elapsed / current) * (total - current)
    tail = label
    if len(tail) > 34:
        tail = f"...{tail[-31:]}"
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} {percent:3d}% {format_eta(eta)} {tail}   "
    )
    sys.stdout.flush()


def compute_target_size(width: int, height: int, downscale_max_side: int) -> tuple[int, int]:
    if downscale_max_side <= 0:
        return width, height
    max_side = max(width, height)
    if max_side <= downscale_max_side:
        return width, height
    scale = downscale_max_side / max_side
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def load_image_for_compare(
    image_path: Path, target_size: tuple[int, int], blur_radius: float
) -> np.ndarray:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img = img.convert("L")
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
    return array


def delete_similar_frames(
    source_dir: Path,
    max_mean_diff: float,
    downscale_max_side: int,
    block_size: int,
    blur_radius: float,
    dry_run: bool,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    print(
        "Starting similar-frame cleanup.\n"
        f"- Source: {source_dir}\n"
        f"- Extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}\n"
        f"- Max mean diff: {max_mean_diff:.4f}\n"
        f"- Downscale max side: {downscale_max_side}\n"
        f"- Block size: {block_size}\n"
        f"- Blur radius: {blur_radius}\n"
        f"- Dry run: {'yes' if dry_run else 'no'}\n"
        f"- HEIC support: {'enabled' if HEIF_SUPPORTED else 'missing'}"
    )

    if not HEIF_SUPPORTED:
        unsupported = any(path.suffix.lower() == ".heic" for path in source_dir.iterdir())
        if unsupported:
            raise ImportError(
                "HEIC support requires the optional pillow-heif package. "
                "Install it with `pip install pillow-heif` and retry."
            )

    files = [
        path
        for path in sorted(source_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    total_files = len(files)
    if total_files:
        print(f"Found {total_files} images. Processing...")
    else:
        print("No images found to process.")
        return

    deleted = 0
    kept = 0
    skipped = 0
    errors = 0
    deleted_bytes = 0
    size_mismatch = 0

    prev_array: np.ndarray | None = None
    prev_size: tuple[int, int] | None = None
    target_size: tuple[int, int] | None = None

    start_time = time.time()

    for index, image_path in enumerate(files, start=1):
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if prev_size and img.size != prev_size:
                    size_mismatch += 1
                prev_size = img.size
                if target_size is None:
                    target_size = compute_target_size(
                        img.size[0], img.size[1], downscale_max_side
                    )
            if target_size is None:
                skipped += 1
                render_progress(index, total_files, start_time, image_path.name)
                continue

            current_array = load_image_for_compare(
                image_path, target_size, blur_radius
            )

            delete_current = False
            if prev_array is not None:
                diff = np.abs(current_array - prev_array)
                height, width = diff.shape
                max_block_mean = 0.0
                for y in range(0, height, block_size):
                    for x in range(0, width, block_size):
                        block = diff[y : y + block_size, x : x + block_size]
                        block_mean = float(np.mean(block))
                        if block_mean > max_block_mean:
                            max_block_mean = block_mean
                delete_current = max_block_mean <= max_mean_diff

            if delete_current:
                deleted += 1
                deleted_bytes += image_path.stat().st_size
                if not dry_run:
                    image_path.unlink()
            else:
                kept += 1

            prev_array = current_array

        except UnidentifiedImageError:
            skipped += 1
        except OSError:
            errors += 1
        finally:
            render_progress(index, total_files, start_time, image_path.name)

    print()
    deleted_bytes_label = "Deleted bytes"
    if dry_run:
        deleted_bytes_label = "Deleted bytes (dry-run estimate)"
    print(
        "Done.\n"
        f"- Images scanned: {total_files}\n"
        f"- Kept: {kept}\n"
        f"- Deleted: {deleted}\n"
        f"- Skipped (not image): {skipped}\n"
        f"- Errors: {errors}\n"
        f"- Size mismatches: {size_mismatch}\n"
        f"- {deleted_bytes_label}: {deleted_bytes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete near-duplicate frames by comparing each image to the previous one."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("images"),
        help="Source directory containing images. Default: './images'",
    )
    parser.add_argument(
        "--max-mean-diff",
        type=float,
        default=0.01,
        help="Delete when mean absolute difference <= this value (0..1). Default: 0.01",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=160,
        help="Downscale max side before comparison. Use 0 to disable. Default: 160",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size (pixels) for local difference comparison. Default: 16",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Gaussian blur radius before comparison. Default: 0.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete files; only report what would be removed.",
    )

    args = parser.parse_args()
    if args.max_mean_diff < 0:
        raise ValueError("--max-mean-diff must be >= 0")
    if args.downscale < 0:
        raise ValueError("--downscale must be >= 0")
    if args.block_size <= 0:
        raise ValueError("--block-size must be > 0")
    if args.blur < 0:
        raise ValueError("--blur must be >= 0")

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")

    source = resolve_path(args.source, cwd)
    delete_similar_frames(
        source,
        args.max_mean_diff,
        args.downscale,
        args.block_size,
        args.blur,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
