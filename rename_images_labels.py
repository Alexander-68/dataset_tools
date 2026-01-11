from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


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


def make_temp_path(directory: Path, index: int, original_name: str) -> Path:
    candidate = directory / f".__rename_tmp__{index}__{original_name}"
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = directory / f".__rename_tmp__{index}_{counter}__{original_name}"
    return candidate


def collect_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_mappings(
    images: list[Path],
    labels_dir: Path | None,
    prefix: str,
    start_index: int,
) -> tuple[int, list[dict[str, Path | None]]]:
    total = len(images)
    width = max(1, len(str(start_index + total - 1))) if total else 1
    mappings: list[dict[str, Path]] = []
    for offset, image_path in enumerate(images):
        seq = start_index + offset
        number = str(seq).zfill(width)
        new_stem = f"{prefix}{number}"
        new_image_path = image_path.with_name(
            f"{new_stem}{image_path.suffix.lower()}"
        )
        label_path = (
            labels_dir / f"{image_path.stem}.txt" if labels_dir is not None else None
        )
        new_label_path = labels_dir / f"{new_stem}.txt" if labels_dir else None
        mappings.append(
            {
                "image_path": image_path,
                "new_image_path": new_image_path,
                "label_path": label_path,
                "new_label_path": new_label_path,
            }
        )
    return width, mappings


def ensure_no_conflicts(mappings: list[dict[str, Path | None]]) -> None:
    source_images = {mapping["image_path"] for mapping in mappings}
    target_images: set[Path] = set()
    for mapping in mappings:
        target = mapping["new_image_path"]
        if target in target_images:
            raise ValueError(f"Duplicate target image name: {target.name}")
        target_images.add(target)
        if target.exists() and target not in source_images:
            raise FileExistsError(f"Target image already exists: {target}")

    source_labels = {
        mapping["label_path"]
        for mapping in mappings
        if mapping["label_path"] is not None and mapping["label_path"].exists()
    }
    target_labels: set[Path] = set()
    for mapping in mappings:
        label_path = mapping["label_path"]
        if label_path is None or not label_path.exists():
            continue
        target = mapping["new_label_path"]
        if target is None:
            continue
        if target in target_labels:
            raise ValueError(f"Duplicate target label name: {target.name}")
        target_labels.add(target)
        if target.exists() and target not in source_labels:
            raise FileExistsError(f"Target label already exists: {target}")


def rename_images_labels(
    images_dir: Path,
    labels_dir: Path | None,
    prefix: str,
    start_index: int,
) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if labels_dir is not None and not labels_dir.is_dir():
        labels_dir = None

    images = collect_images(images_dir)
    total = len(images)
    width, mappings = build_mappings(images, labels_dir, prefix, start_index)

    labels_display = labels_dir if labels_dir is not None else "<disabled>"
    print(
        "Renaming images and labels.\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_display}\n"
        f"- Prefix: {prefix if prefix else '<none>'}\n"
        f"- Start index: {start_index}\n"
        f"- Zero pad width: {width}\n"
        f"- Image count: {total}"
    )

    if total == 0:
        print("No images found to rename.")
        return

    ensure_no_conflicts(mappings)

    temp_image_map: dict[Path, Path] = {}
    temp_label_map: dict[Path, Path] = {}
    missing_labels = 0

    for index, mapping in enumerate(mappings, start=1):
        image_path = mapping["image_path"]
        target_image = mapping["new_image_path"]
        if image_path != target_image:
            temp_path = make_temp_path(images_dir, index, image_path.name)
            image_path.rename(temp_path)
            temp_image_map[target_image] = temp_path

        label_path = mapping["label_path"]
        if label_path is not None:
            if label_path.exists():
                target_label = mapping["new_label_path"]
                if target_label is not None and label_path != target_label:
                    temp_label = make_temp_path(labels_dir, index, label_path.name)
                    label_path.rename(temp_label)
                    temp_label_map[target_label] = temp_label
            else:
                missing_labels += 1

    renamed_images = 0
    renamed_labels = 0
    skipped_images = 0

    for index, mapping in enumerate(mappings, start=1):
        target_image = mapping["new_image_path"]
        if target_image in temp_image_map:
            temp_image_map[target_image].rename(target_image)
            renamed_images += 1
        else:
            skipped_images += 1

        target_label = mapping["new_label_path"]
        if target_label is not None and target_label in temp_label_map:
            temp_label_map[target_label].rename(target_label)
            renamed_labels += 1

        render_progress(index, total, target_image.name)

    print()
    print(
        "Done.\n"
        f"- Images processed: {total}\n"
        f"- Images renamed: {renamed_images}\n"
        f"- Images skipped: {skipped_images}\n"
        f"- Labels renamed: {renamed_labels}\n"
        f"- Labels missing: {missing_labels}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename images and corresponding label files sequentially."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Images directory (relative to CWD). Defaults to './images' if it exists.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Labels directory (relative to CWD). Defaults to './labels' if it exists."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for new filenames (e.g., 'train_').",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index for the sequence. Default: 1",
    )

    args = parser.parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_arg = args.images if args.images is not None else Path("images")
    labels_arg = args.labels if args.labels is not None else Path("labels")
    images_dir = resolve_path(images_arg, cwd)
    labels_dir = resolve_path(labels_arg, cwd) if labels_arg is not None else None

    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    rename_images_labels(images_dir, labels_dir, args.prefix, args.start)


if __name__ == "__main__":
    main()
