from __future__ import annotations

import argparse
import math
import time
from pathlib import Path


FACE_INDICES = [0, 1, 2, 3, 4]
FACE_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
NOSE_INDEX = 0
EYE_EAR_INDICES = [1, 2, 3, 4]
EYE_EAR_NAMES = ["left_eye", "right_eye", "left_ear", "right_ear"]


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress(current: int, total: int, start_time: float, name: str) -> None:
    if total <= 0:
        return
    elapsed = max(0.0001, time.time() - start_time)
    rate = current / elapsed if current else 0.0
    eta_seconds = (total - current) / rate if rate else None
    eta = format_eta(eta_seconds)
    bar_width = 28
    filled = int(round(bar_width * current / total))
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(100 * current / total)
    tail = name
    if len(tail) > 36:
        tail = f"...{tail[-33:]}"
    print(f"\r[{bar}] {current}/{total} {percent:3d}% ETA {eta} {tail}   ", end="", flush=True)


def parse_yolo_pose(line: str) -> dict | None:
    parts = line.strip().split()
    if not parts:
        return None
    class_id = int(float(parts[0]))
    if len(parts) < 5:
        return {"class_id": class_id, "bbox": [], "kpts": []}
    bbox = [float(v) for v in parts[1:5]]
    kpt_vals = [float(v) for v in parts[5:]]
    kpts = []
    for idx in range(0, len(kpt_vals), 3):
        if idx + 2 >= len(kpt_vals):
            break
        kpts.append([kpt_vals[idx], kpt_vals[idx + 1], kpt_vals[idx + 2]])
    return {"class_id": class_id, "bbox": bbox, "kpts": kpts}


def format_yolo_pose(class_id: int, bbox: list[float], kpts: list[list[float]]) -> str:
    parts = [str(class_id)]
    if bbox:
        parts.extend(f"{v:.6f}" for v in bbox)
    for kp in kpts:
        parts.append(f"{kp[0]:.6f}")
        parts.append(f"{kp[1]:.6f}")
        parts.append(f"{kp[2]:.6f}")
    return " ".join(parts)


def is_valid_kpt(kp: list[float]) -> bool:
    return (kp[2] > 0) and not (kp[0] == 0 and kp[1] == 0)


def get_nose(kpts: list[list[float]]) -> tuple[float, float] | None:
    if NOSE_INDEX >= len(kpts):
        return None
    kp = kpts[NOSE_INDEX]
    if not is_valid_kpt(kp):
        return None
    return (kp[0], kp[1])


def get_face_center(kpts: list[list[float]]) -> tuple[float, float] | None:
    xs = []
    ys = []
    for idx in FACE_INDICES:
        if idx >= len(kpts):
            continue
        kp = kpts[idx]
        if is_valid_kpt(kp):
            xs.append(kp[0])
            ys.append(kp[1])
    if not xs:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)


def merge_eye_ear_kpts(
    base_kpts: list[list[float]],
    face_kpts: list[list[float]],
) -> tuple[list[list[float]], int]:
    merged = [kp[:] for kp in base_kpts]
    min_len = max(EYE_EAR_INDICES) + 1
    if len(merged) < min_len:
        merged.extend([[0.0, 0.0, 0.0] for _ in range(min_len - len(merged))])

    copied = 0
    for idx in EYE_EAR_INDICES:
        if idx >= len(face_kpts):
            continue
        face_kp = face_kpts[idx]
        if is_valid_kpt(face_kp):
            merged[idx] = face_kp[:]
            copied += 1
    return merged, copied


def merge_face_kpts(
    base_kpts: list[list[float]],
    face_kpts: list[list[float]],
) -> tuple[list[list[float]], int]:
    merged = [kp[:] for kp in base_kpts]
    min_len = max(FACE_INDICES) + 1
    if len(merged) < min_len:
        merged.extend([[0.0, 0.0, 0.0] for _ in range(min_len - len(merged))])

    copied = 0
    for idx in FACE_INDICES:
        if idx >= len(face_kpts):
            continue
        face_kp = face_kpts[idx]
        merged[idx] = face_kp[:]
        copied += 1
    return merged, copied


def process_files(
    labels_dir: Path,
    labels_face_dir: Path,
    output_dir: Path,
    use_nose: bool,
    max_dist: float | None,
) -> None:
    print("Starting face keypoint merge.")
    match_desc = "closest nose keypoint (--nose)" if use_nose else "closest face area center"
    if max_dist is not None:
        match_desc = f"{match_desc} within {max_dist:.3f}"
    print(
        "Parameters:\n"
        f"- Labels: {labels_dir}\n"
        f"- Labels face: {labels_face_dir}\n"
        f"- Output: {output_dir}\n"
        f"- Match: {match_desc}"
    )

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    if not labels_face_dir.is_dir():
        raise FileNotFoundError(f"labels-face directory not found: {labels_face_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    total_files = len(label_files)
    print(f"Found {total_files} label files in '{labels_dir}'.")

    total_persons = 0
    matched_persons = 0
    unmatched_persons = 0
    missing_nose_base = 0
    missing_nose_face = 0
    missing_face_base = 0
    missing_face_face = 0
    eyes_ears_copied = 0
    face_kpts_copied = 0
    match_dist_sum = 0.0
    match_dist_count = 0

    start_time = time.time()
    for index, base_path in enumerate(label_files, start=1):
        face_path = labels_face_dir / base_path.name
        output_path = output_dir / base_path.name

        base_entries = []
        with base_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = parse_yolo_pose(line)
                if entry is not None:
                    base_entries.append(entry)

        face_entries = []
        if labels_face_dir.is_dir() and face_path.exists():
            with face_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = parse_yolo_pose(line)
                    if entry is not None:
                        face_entries.append(entry)

        face_people = [entry for entry in face_entries if entry["class_id"] == 0 and entry["kpts"]]
        available_faces = set(range(len(face_people)))

        face_noses: dict[int, tuple[float, float]] = {}
        face_centers: dict[int, tuple[float, float]] = {}
        for idx, entry in enumerate(face_people):
            if use_nose:
                nose = get_nose(entry["kpts"])
                if nose is not None:
                    face_noses[idx] = nose
            else:
                center = get_face_center(entry["kpts"])
                if center is not None:
                    face_centers[idx] = center

        output_lines = []
        for entry in base_entries:
            if entry["class_id"] != 0 or not entry["kpts"]:
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            total_persons += 1
            if use_nose:
                base_anchor = get_nose(entry["kpts"])
                if base_anchor is None:
                    missing_nose_base += 1
                    output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                    continue
                candidates = [idx for idx in available_faces if idx in face_noses]
            else:
                base_anchor = get_face_center(entry["kpts"])
                if base_anchor is None:
                    missing_face_base += 1
                    output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                    continue
                candidates = [idx for idx in available_faces if idx in face_centers]

            if not candidates:
                unmatched_persons += 1
                if face_people:
                    if use_nose:
                        missing_nose_face += 1
                    else:
                        missing_face_face += 1
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            best_idx = None
            best_dist = None
            for idx in candidates:
                face_anchor = face_noses[idx] if use_nose else face_centers[idx]
                dist = math.hypot(base_anchor[0] - face_anchor[0], base_anchor[1] - face_anchor[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is None:
                unmatched_persons += 1
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue
            if max_dist is not None and best_dist is not None and best_dist > max_dist:
                unmatched_persons += 1
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            matched_persons += 1
            if best_dist is not None:
                match_dist_sum += best_dist
                match_dist_count += 1
            available_faces.discard(best_idx)

            face_entry = face_people[best_idx]
            if use_nose:
                merged_kpts, copied = merge_eye_ear_kpts(entry["kpts"], face_entry["kpts"])
                eyes_ears_copied += copied
            else:
                merged_kpts, copied = merge_face_kpts(entry["kpts"], face_entry["kpts"])
                face_kpts_copied += copied
            output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], merged_kpts))

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(output_lines))

        render_progress(index, total_files, start_time, base_path.name)

    if total_files:
        print()

    print("Finished face keypoint merge.")
    print(f"Files processed: {total_files}")
    print(f"Total persons: {total_persons}")
    print(f"Matched persons: {matched_persons}")
    print(f"Unmatched persons: {unmatched_persons}")
    if use_nose:
        print(f"Missing nose in base: {missing_nose_base}")
        print(f"Missing nose in face: {missing_nose_face}")
        print(f"Eyes/ears copied: {eyes_ears_copied}")
    else:
        print(f"Missing face area in base: {missing_face_base}")
        print(f"Missing face area in face: {missing_face_face}")
        print(f"Face keypoints copied: {face_kpts_copied}")
    if match_dist_count:
        avg_dist = match_dist_sum / match_dist_count
        label = "Average nose distance" if use_nose else "Average face-center distance"
        print(f"{label}: {avg_dist:.6f} (n={match_dist_count})")
    else:
        label = "Average nose distance" if use_nose else "Average face-center distance"
        print(f"{label}: n/a (n=0)")
    updated_names = EYE_EAR_NAMES if use_nose else FACE_NAMES
    print("Updated keypoints: " + ", ".join(updated_names))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add face keypoints to YOLO pose labels using closest face-area matching "
            "(use --nose for closest-nose matching)."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels"),
        help="Folder with base pose annotations (default: labels).",
    )
    parser.add_argument(
        "--labels-face",
        type=Path,
        default=Path("labels-face"),
        help="Folder with face pose annotations (default: labels-face).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labels-merged"),
        help="Output folder for merged labels (default: labels-merged).",
    )
    parser.add_argument(
        "--nose",
        action="store_true",
        help="Use legacy closest-nose matching and only update eye/ear keypoints.",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=None,
        help="Maximum normalized distance for a match (default: no limit).",
    )

    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    labels_dir = resolve_path(args.labels, cwd)
    labels_face_dir = resolve_path(args.labels_face, cwd)
    output_dir = resolve_path(args.output, cwd)

    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")

    process_files(labels_dir, labels_face_dir, output_dir, args.nose, args.max_dist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
