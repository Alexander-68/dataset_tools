from __future__ import annotations

import shutil
from pathlib import Path
from collections import defaultdict


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


def render_progress(current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def count_instances(label_path: Path) -> int:
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        raise OSError(f"Could not read label file {label_path}: {exc}") from exc


def merge_datasets(root: Path) -> None:
    verified_dir = root / "Verified"
    target_dir = verified_dir / "Final_dataset"

    print(
        "Merging verified datasets.\n"
        f"- Data root (CWD): {root}\n"
        f"- Verified dir: {verified_dir}\n"
        f"- Target dir: {target_dir}"
    )

    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target dataset directory not found: {target_dir}")

    images_train_dir = target_dir / "images" / "train"
    images_val_dir = target_dir / "images" / "val"
    labels_train_dir = target_dir / "labels" / "train"
    labels_val_dir = target_dir / "labels" / "val"

    for path in (images_train_dir, images_val_dir, labels_train_dir, labels_val_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_dirs = [
        path
        for path in sorted(verified_dir.iterdir())
        if path.is_dir() and path != target_dir
    ]

    if not source_dirs:
        print("No source dataset folders found to merge.")
        return

    source_entries: list[tuple[Path, Path, Path, list[Path]]] = []
    total_images = 0
    for source in source_dirs:
        images_dir = source / "images"
        labels_dir = source / "labels"

        if not images_dir.is_dir():
            print(f"Skipping {source}: missing images directory.")
            continue
        if not labels_dir.is_dir():
            print(f"Skipping {source}: missing labels directory.")
            continue

        image_paths = [
            path
            for path in sorted(images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        source_entries.append((source, images_dir, labels_dir, image_paths))
        total_images += len(image_paths)

    if not source_entries:
        print("No valid source dataset folders found to merge.")
        return

    print(f"Total images queued: {total_images}")

    processed = 0
    copied = 0
    progress_count = 0
    source_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "train_images": 0,
            "val_images": 0,
            "train_instances": 0,
            "val_instances": 0,
        }
    )

    for source, images_dir, labels_dir, image_paths in source_entries:
        validation_enabled = len(image_paths) >= 64

        for image_path in image_paths:

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Label file not found for image {image_path.name} in {labels_dir}"
                )

            instance_count = count_instances(label_path)
            if validation_enabled:
                processed += 1
                is_validation = processed % 8 == 0
            else:
                is_validation = False
            split_dir_pair = (
                (images_val_dir, labels_val_dir)
                if is_validation
                else (images_train_dir, labels_train_dir)
            )

            image_dest_dir, label_dest_dir = split_dir_pair
            image_dest_name = f"{source.name}_{image_path.name}"
            label_dest_name = f"{source.name}_{image_path.stem}.txt"
            image_dest_path = image_dest_dir / image_dest_name
            label_dest_path = label_dest_dir / label_dest_name

            if image_dest_path.exists() or label_dest_path.exists():
                raise FileExistsError(
                    f"Destination files already exist: {image_dest_path} / {label_dest_path}"
                )

            shutil.copy2(image_path, image_dest_path)
            shutil.copy2(label_path, label_dest_path)
            copied += 1
            split_key = "val" if is_validation else "train"
            source_stat = source_stats[source.name]
            image_key = f"{split_key}_images"
            instance_key = f"{split_key}_instances"
            source_stat[image_key] += 1
            source_stat[instance_key] += instance_count
            progress_count += 1
            render_progress(progress_count, total_images)

    if total_images:
        print()

    write_content_md(target_dir, dict(sorted(source_stats.items())))

    print(
        f"Merged {copied} image/label pairs from {len(source_entries)} source folders into "
        f"{target_dir} (every 8th pair in validation; sources with <64 images go to train only)."
    )


def write_content_md(target_dir: Path, stats: dict[str, dict[str, int]]) -> None:
    content_path = target_dir / "content.md"
    total_train_images = sum(values.get("train_images", 0) for values in stats.values())
    total_val_images = sum(values.get("val_images", 0) for values in stats.values())
    total_images = total_train_images + total_val_images
    total_train_instances = sum(
        values.get("train_instances", 0) for values in stats.values()
    )
    total_val_instances = sum(
        values.get("val_instances", 0) for values in stats.values()
    )
    total_instances = total_train_instances + total_val_instances

    summary_lines = ["# Merge Summary", "", "Sources merged:"]
    if stats:
        for source, values in stats.items():
            train_images = values.get("train_images", 0)
            val_images = values.get("val_images", 0)
            source_total_images = train_images + val_images
            train_instances = values.get("train_instances", 0)
            val_instances = values.get("val_instances", 0)
            source_total_instances = train_instances + val_instances
            summary_lines.append(
                f"- {source}: {source_total_images} images "
                f"(train {train_images}, val {val_images}); "
                f"{source_total_instances} instances "
                f"(train {train_instances}, val {val_instances})"
            )
    else:
        summary_lines.append("- None (no new sources merged)")
    summary_lines.append("")
    summary_lines.append(
        "Totals: "
        f"{total_images} images (train {total_train_images}, val {total_val_images}); "
        f"{total_instances} instances "
        f"(train {total_train_instances}, val {total_val_instances})"
    )

    try:
        content_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not write content file {content_path}: {exc}")


def main() -> int:
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    print(f"Script dir: {script_dir}")
    merge_datasets(cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
