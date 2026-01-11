from pathlib import Path
import shutil


def render_progress(current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def main() -> int:
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    labels_dir = cwd / "labels"
    images_dir = cwd / "images"
    target_dir = cwd / "labels-x"

    print(
        "Cleaning up labels without matching images.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Target dir: {target_dir}"
    )

    if not labels_dir.is_dir():
        print(f"Error: Labels directory not found: {labels_dir}")
        return 1
    if not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)

    label_files = [p for p in labels_dir.iterdir() if p.is_file()]
    image_base_names = {p.stem for p in images_dir.iterdir() if p.is_file()}

    moved_count = 0
    for idx, label_file in enumerate(label_files, start=1):
        if label_file.suffix.lower() != ".txt":
            render_progress(idx, len(label_files))
            continue

        if label_file.stem not in image_base_names:
            dst_path = target_dir / label_file.name
            print(f"Moving {label_file.name} to {target_dir}")
            shutil.move(str(label_file), str(dst_path))
            moved_count += 1
        render_progress(idx, len(label_files))

    if label_files:
        print()
    print(f"Moved {moved_count} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
