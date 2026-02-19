from __future__ import annotations

import argparse
import math
import shutil
import time
from datetime import datetime, timedelta
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


def render_progress(current: int, total: int, start_time: float, label: str) -> None:
    if total <= 0:
        return
    elapsed = max(0.0001, time.time() - start_time)
    rate = current / elapsed if current else 0.0
    eta_label = "ETA --:--"
    if rate > 0:
        eta_seconds = (total - current) / rate
        if math.isfinite(eta_seconds):
            eta_time = datetime.now() + timedelta(seconds=max(0.0, eta_seconds))
            eta_label = f"ETA {eta_time.strftime('%H:%M')}"

    width = 28
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    percent = int(100 * current / total)
    tail = label
    if len(tail) > 36:
        tail = f"...{tail[-33:]}"
    print(
        f"\r[{bar}] {current}/{total} {percent:3d}% {eta_label} {tail}   ",
        end="",
        flush=True,
    )


def collect_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def collect_label_txt(labels_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(labels_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    ]


def run_cleanup(images_dir: Path, labels_dir: Path, labels_x_dir: Path) -> int:
    if not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_x_dir.mkdir(parents=True, exist_ok=True)

    image_files_before = collect_images(images_dir)
    label_files_before = collect_label_txt(labels_dir)
    image_files_for_rename = [path for path in sorted(images_dir.iterdir()) if path.is_file()]
    label_files_for_rename = [path for path in sorted(labels_dir.iterdir()) if path.is_file()]
    rename_total = len(image_files_for_rename) + len(label_files_for_rename)

    print(
        "Cleanup labels run.\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Orphan labels target dir: {labels_x_dir}\n"
        f"- Image extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}\n"
        f"- Image files found before rename: {len(image_files_before)}\n"
        f"- Label files found before rename: {len(label_files_before)}\n"
        f"- Filename normalization ops: {rename_total}"
    )

    renamed_images = 0
    renamed_labels = 0
    rename_conflicts_images = 0
    rename_conflicts_labels = 0

    rename_start_time = time.time()
    rename_step = 0

    for path in image_files_for_rename:
        rename_step += 1
        if " " in path.name:
            target = path.with_name(path.name.replace(" ", "_"))
            if target.exists() and target != path:
                print(
                    f"\nWarning: cannot rename due to existing target: "
                    f"{path.name} -> {target.name}"
                )
                rename_conflicts_images += 1
            else:
                path.rename(target)
                renamed_images += 1
        render_progress(rename_step, rename_total, rename_start_time, path.name)

    for path in label_files_for_rename:
        rename_step += 1
        if " " in path.name:
            target = path.with_name(path.name.replace(" ", "_"))
            if target.exists() and target != path:
                print(
                    f"\nWarning: cannot rename due to existing target: "
                    f"{path.name} -> {target.name}"
                )
                rename_conflicts_labels += 1
            else:
                path.rename(target)
                renamed_labels += 1
        render_progress(rename_step, rename_total, rename_start_time, path.name)

    if rename_total:
        print()

    image_files = collect_images(images_dir)
    label_files = collect_label_txt(labels_dir)
    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}
    cleanup_total = len(label_files) + len(image_files)

    moved_orphan_labels = 0
    created_empty_labels = 0
    skipped_existing_labels = 0

    cleanup_start_time = time.time()
    cleanup_step = 0

    for label_path in label_files:
        cleanup_step += 1
        if label_path.stem not in image_stems:
            destination = labels_x_dir / label_path.name
            shutil.move(str(label_path), str(destination))
            moved_orphan_labels += 1
        render_progress(cleanup_step, cleanup_total, cleanup_start_time, label_path.name)

    for image_path in image_files:
        cleanup_step += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        if image_path.stem not in label_stems and not label_path.exists():
            label_path.write_text("", encoding="utf-8")
            created_empty_labels += 1
        else:
            skipped_existing_labels += 1
        render_progress(cleanup_step, cleanup_total, cleanup_start_time, image_path.name)

    if cleanup_total:
        print()

    print(
        "Done.\n"
        f"- Image files found after rename: {len(image_files)}\n"
        f"- Label files found after rename: {len(label_files)}\n"
        f"- Images scanned: {len(image_files)}\n"
        f"- Labels scanned: {len(label_files)}\n"
        f"- Filenames renamed in images/: {renamed_images}\n"
        f"- Filenames renamed in labels/: {renamed_labels}\n"
        f"- Rename conflicts in images/: {rename_conflicts_images}\n"
        f"- Rename conflicts in labels/: {rename_conflicts_labels}\n"
        f"- Orphan label files moved to labels-x: {moved_orphan_labels}\n"
        f"- Empty label files created: {created_empty_labels}\n"
        f"- Images already having labels: {skipped_existing_labels}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize filenames (spaces->underscores) in images/labels, move "
            "orphan labels to labels-x, and create empty label files for "
            "unlabeled images."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images"),
        help="Images directory (relative to CWD). Default: ./images",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels"),
        help="Labels directory (relative to CWD). Default: ./labels",
    )
    parser.add_argument(
        "--labels-x",
        type=Path,
        default=Path("labels-x"),
        help="Target directory for orphan label files. Default: ./labels-x",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(args.images, cwd)
    labels_dir = resolve_path(args.labels, cwd)
    labels_x_dir = resolve_path(args.labels_x, cwd)

    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    return run_cleanup(images_dir, labels_dir, labels_x_dir)


if __name__ == "__main__":
    raise SystemExit(main())
