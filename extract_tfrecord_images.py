from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None

IMAGE_KEYS = ("image/encoded", "image", "image_raw", "jpeg", "png", "encoded")
FILENAME_KEYS = ("image/filename", "filename", "file_name", "image/name")


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


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def get_filename(features, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        feature = features.get(key)
        if feature and feature.bytes_list.value:
            try:
                return feature.bytes_list.value[0].decode("utf-8", errors="ignore")
            except Exception:
                return None
    return None


def bytes_candidates(features) -> list[tuple[str, bytes]]:
    candidates = []
    for key, feature in features.items():
        if feature.bytes_list.value:
            candidates.append((key, feature.bytes_list.value[0]))
    return candidates


def looks_like_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def find_image_bytes(features, preferred_key: Optional[str]) -> tuple[Optional[bytes], Optional[str]]:
    candidates = bytes_candidates(features)
    if preferred_key:
        for key, data in candidates:
            if key == preferred_key:
                return data, key
        return None, None

    for key in IMAGE_KEYS:
        for cand_key, data in candidates:
            if cand_key == key:
                return data, cand_key

    for cand_key, data in candidates:
        if cand_key in FILENAME_KEYS:
            continue
        if looks_like_image(data):
            return data, cand_key

    return None, None


def safe_output_path(output_dir: Path, filename: Optional[str], index: int) -> Path:
    if filename:
        name = Path(filename).name
        stem = Path(name).stem or f"image_{index:06d}"
    else:
        stem = f"image_{index:06d}"
    output_path = output_dir / f"{stem}.jpg"
    if not output_path.exists():
        return output_path
    counter = 1
    while True:
        candidate = output_dir / f"{stem}_{counter:02d}.jpg"
        if not candidate.exists():
            return candidate
        counter += 1


def count_records(tfrecord_path: Path) -> int:
    dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
    total = 0
    for _ in dataset:
        total += 1
    return total


def extract_images(
    tfrecord_path: Path,
    output_dir: Path,
    quality: int,
    image_key: Optional[str],
    filename_key: Optional[str],
) -> None:
    if not tfrecord_path.is_file():
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_records = count_records(tfrecord_path)
    if total_records:
        print(f"Found {total_records} records. Extracting...")
    else:
        print("No records found in TFRecord.")
        return

    extracted = 0
    skipped = 0
    invalid = 0
    used_keys: dict[str, int] = {}

    dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
    start_time = time.monotonic()
    for index, record in enumerate(dataset, start=1):
        label = f"record {index}"
        try:
            example = tf.train.Example.FromString(bytes(record.numpy()))
        except Exception:
            invalid += 1
            render_progress(index, total_records, label, start_time)
            continue

        features = example.features.feature
        img_bytes, used_key = find_image_bytes(features, image_key)
        if img_bytes is None:
            skipped += 1
            render_progress(index, total_records, label, start_time)
            continue

        filename = None
        if filename_key:
            filename = get_filename(features, (filename_key,))
        if filename is None:
            filename = get_filename(features, FILENAME_KEYS)

        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                output_path = safe_output_path(output_dir, filename, index)
                img.save(output_path, format="JPEG", quality=quality)
            extracted += 1
            if used_key:
                used_keys[used_key] = used_keys.get(used_key, 0) + 1
        except (UnidentifiedImageError, OSError, ValueError):
            invalid += 1
        finally:
            render_progress(index, total_records, label, start_time)

    if total_records:
        print()

    print(
        "Extraction complete.\n"
        f"- Records scanned: {total_records}\n"
        f"- Images written: {extracted}\n"
        f"- Records skipped (no image): {skipped}\n"
        f"- Records failed (decode/save): {invalid}"
    )
    if used_keys:
        ordered = ", ".join(
            f"{key}={count}" for key, count in sorted(used_keys.items())
        )
        print(f"- Image keys used: {ordered}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract images from a TFRecord file into JPEGs."
    )
    parser.add_argument(
        "tfrecord",
        type=Path,
        help="Path to the TFRecord (.tfrec/.tfrecord) file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images"),
        help="Output directory for extracted images. Default: './images'",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100). Default: 95",
    )
    parser.add_argument(
        "--image-key",
        default=None,
        help="Override image feature key (e.g. 'image/encoded').",
    )
    parser.add_argument(
        "--filename-key",
        default=None,
        help="Override filename feature key (e.g. 'image/filename').",
    )

    args = parser.parse_args()
    if tf is None:
        raise ImportError(
            "TensorFlow is required to read TFRecord files. Install it with "
            "`pip install tensorflow` and retry."
        )

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    tfrecord_path = resolve_path(args.tfrecord, cwd)
    output_dir = resolve_path(args.output, cwd)
    quality = max(1, min(100, int(args.quality)))

    print(
        "Extracting images from TFRecord.\n"
        f"- TFRecord: {tfrecord_path}\n"
        f"- Output: {output_dir}\n"
        f"- JPEG quality: {quality}\n"
        f"- Image key override: {args.image_key or 'auto'}\n"
        f"- Filename key override: {args.filename_key or 'auto'}"
    )
    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")

    extract_images(
        tfrecord_path=tfrecord_path,
        output_dir=output_dir,
        quality=quality,
        image_key=args.image_key,
        filename_key=args.filename_key,
    )


if __name__ == "__main__":
    main()
