from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency
    HEIF_SUPPORTED = False
else:
    register_heif_opener()
    HEIF_SUPPORTED = True


HEIC_EXTENSIONS = {".heic", ".heif"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} PB"


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


def resize_images(source_dir: Path, dest_dir: Path | None, max_dim: int) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    mode = "in-place" if dest_dir is None else f"destination: {dest_dir}"
    print(
        "Starting resize.\n"
        f"- Source: {source_dir}\n"
        f"- Mode: {mode}\n"
        f"- Max dimension: {max_dim}\n"
        f"- HEIC support: {'enabled' if HEIF_SUPPORTED else 'missing'}"
    )

    if dest_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)

    if not HEIF_SUPPORTED:
        unsupported = any(path.suffix.lower() in HEIC_EXTENSIONS for path in source_dir.iterdir())
        if unsupported:
            raise ImportError(
                "HEIC support requires the optional pillow-heif package. "
                "Install it with `pip install pillow-heif` and retry."
            )

    processed = 0
    resized = 0
    converted = 0
    skipped = 0
    total_before = 0
    total_after = 0

    files = [path for path in sorted(source_dir.iterdir()) if path.is_file()]
    total_files = len(files)
    if total_files:
        print(f"Found {total_files} files. Processing...")
    else:
        print("No files found to process.")

    for index, image_path in enumerate(files, start=1):
        if not image_path.is_file():
            continue

        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size

                max_side = max(width, height)
                needs_resize = max_side > max_dim
                scale = max_side / max_dim if needs_resize else 1.0
                
                suffix = image_path.suffix.lower()
                is_heic = suffix in HEIC_EXTENSIONS

                # If in-place and no changes needed, skip
                if not dest_dir and not needs_resize and not is_heic:
                    processed += 1
                    before_size = image_path.stat().st_size
                    total_before += before_size
                    total_after += before_size
                    render_progress(index, total_files, image_path.name)
                    continue

                processed += 1
                total_before += image_path.stat().st_size

                new_width = int(round(width / scale))
                new_height = int(round(height / scale))

                if needs_resize:
                    working_img = img.resize((new_width, new_height), Image.LANCZOS)
                else:
                    working_img = img

                # Determine output path
                if dest_dir:
                    fname = image_path.name
                    if is_heic:
                        fname = image_path.with_suffix(".jpg").name
                    output_path = dest_dir / fname
                else:
                    output_path = image_path.with_suffix(".jpg") if is_heic else image_path

                output_suffix = output_path.suffix.lower()

                save_img = working_img
                if output_suffix in JPEG_EXTENSIONS and save_img.mode not in ("RGB", "L"):
                    save_img = save_img.convert("RGB")

                save_kwargs = {"quality": 95}
                if output_suffix in JPEG_EXTENSIONS:
                    save_kwargs["format"] = "JPEG"

                save_img.save(output_path, **save_kwargs)

                total_after += output_path.stat().st_size

                # Cleanup for in-place HEIC conversion
                if not dest_dir and is_heic and output_path != image_path and image_path.exists():
                    image_path.unlink()

                if needs_resize:
                    resized += 1
                if is_heic:
                    converted += 1

        except (UnidentifiedImageError, OSError):
            # Not an image or cannot be opened
            skipped += 1
        finally:
            render_progress(index, total_files, image_path.name)

    if total_files:
        print()

    message = f"Scanned {processed + skipped} files. Processed {processed} images."
    if resized:
        message += f" Resized {resized} images to max dimension {max_dim}."
    if converted:
        message += f" Converted {converted} HEIC images to JPEG."
    if dest_dir:
        message += f" Saved to {dest_dir}."
    print(message)
    print(
        "Size stats (processed images): "
        f"before {format_bytes(total_before)} -> after {format_bytes(total_after)}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize images in a directory.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("images"),
        help="Source directory containing images. Default: './images'",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Destination directory. If not provided, modifies images in-place.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Maximum dimension for resizing. Default: 1024",
    )

    args = parser.parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    source = resolve_path(args.source, cwd)
    destination = (
        resolve_path(args.destination, cwd) if args.destination is not None else None
    )
    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    resize_images(source, destination, args.size)


if __name__ == "__main__":
    main()
