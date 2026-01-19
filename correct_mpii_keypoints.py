from __future__ import annotations

import argparse
import math
import time
from pathlib import Path


COCO_KPT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

MPII_TO_COCO = [
    (0, 16, "right_ankle"),
    (1, 14, "right_knee"),
    (2, 12, "right_hip"),
    (3, 11, "left_hip"),
    (4, 13, "left_knee"),
    (5, 15, "left_ankle"),
    (10, 10, "right_wrist"),
    (11, 8, "right_elbow"),
    (12, 6, "right_shoulder"),
    (13, 5, "left_shoulder"),
    (14, 7, "left_elbow"),
    (15, 9, "left_wrist"),
]


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
    bbox = []
    if len(parts) >= 5:
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


def yolo_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float] | None:
    if len(bbox) < 4:
        return None
    cx, cy, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    half_w = w / 2.0
    half_h = h / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def is_valid_kpt(kp: list[float]) -> bool:
    return (kp[2] > 0) and not (kp[0] == 0 and kp[1] == 0)


def inside_ratio(bbox_xyxy: tuple[float, float, float, float], kpts: list[list[float]]) -> tuple[float, int, int]:
    min_x, min_y, max_x, max_y = bbox_xyxy
    valid = 0
    inside = 0
    for kp in kpts:
        if not is_valid_kpt(kp):
            continue
        valid += 1
        if min_x <= kp[0] <= max_x and min_y <= kp[1] <= max_y:
            inside += 1
    if valid == 0:
        return 0.0, 0, 0
    return inside / valid, inside, valid


def match_mpii_person(
    bbox_xyxy: tuple[float, float, float, float],
    mpii_people: list[dict],
    available: set[int],
    threshold: float,
) -> tuple[int | None, float, int, int]:
    best_idx = None
    best_ratio = 0.0
    best_inside = 0
    best_valid = 0
    for idx in available:
        ratio, inside, valid = inside_ratio(bbox_xyxy, mpii_people[idx]["kpts"])
        if valid == 0:
            continue
        if ratio > best_ratio or (math.isclose(ratio, best_ratio) and inside > best_inside):
            best_ratio = ratio
            best_idx = idx
            best_inside = inside
            best_valid = valid
    if best_idx is None or best_ratio < threshold:
        return None, 0.0, 0, 0
    return best_idx, best_ratio, best_inside, best_valid


def replace_keypoints(
    coco_kpts: list[list[float]], mpii_kpts: list[list[float]], replaced_counts: dict[str, int]
) -> int:
    replaced = 0
    for mpii_idx, coco_idx, name in MPII_TO_COCO:
        if coco_idx >= len(coco_kpts) or mpii_idx >= len(mpii_kpts):
            continue
        coco_kpts[coco_idx] = mpii_kpts[mpii_idx][:]
        replaced_counts[name] += 1
        replaced += 1
    return replaced


def drop_keypoints_outside_bbox(
    kpts: list[list[float]], bbox_xyxy: tuple[float, float, float, float] | None
) -> int:
    if bbox_xyxy is None:
        return 0
    min_x, min_y, max_x, max_y = bbox_xyxy
    removed = 0
    for kp in kpts:
        if not is_valid_kpt(kp):
            continue
        if kp[0] < min_x or kp[0] > max_x or kp[1] < min_y or kp[1] > max_y:
            kp[0] = 0.0
            kp[1] = 0.0
            kp[2] = 0.0
            removed += 1
    return removed


def process_files(
    labels_x: Path,
    labels_mpii: Path,
    output_dir: Path,
    match_threshold: float,
) -> None:
    print("Starting COCO keypoint correction using MPII annotations.")
    print(
        "Parameters:\n"
        f"- Labels-x: {labels_x}\n"
        f"- Labels-mpii: {labels_mpii}\n"
        f"- Output: {output_dir}\n"
        f"- Match threshold: {match_threshold}"
    )

    if not labels_x.is_dir():
        raise FileNotFoundError(f"labels-x directory not found: {labels_x}")
    if not labels_mpii.is_dir():
        print(f"Warning: labels-mpii directory not found: {labels_mpii}. Using originals only.")

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_x.glob("*.txt"))
    total_files = len(label_files)
    print(f"Found {total_files} label files in '{labels_x}'.")

    total_people = 0
    matched_people = 0
    unmatched_people = 0
    replaced_keypoints = 0
    removed_keypoints = 0
    replaced_counts = {name: 0 for name in (entry[2] for entry in MPII_TO_COCO)}
    missing_mpii_files = 0

    start_time = time.time()
    for index, label_path in enumerate(label_files, start=1):
        mpii_path = labels_mpii / label_path.name
        if not labels_mpii.is_dir() or not mpii_path.exists():
            missing_mpii_files += 1

        orig_entries: list[dict] = []
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = parse_yolo_pose(line)
                if entry is not None:
                    orig_entries.append(entry)

        mpii_entries: list[dict] = []
        if labels_mpii.is_dir() and mpii_path.exists():
            with mpii_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = parse_yolo_pose(line)
                    if entry is not None:
                        mpii_entries.append(entry)

        mpii_people = [entry for entry in mpii_entries if entry["class_id"] == 0]
        available = set(range(len(mpii_people)))

        output_lines = []
        for entry in orig_entries:
            if entry["class_id"] != 0 or not entry["kpts"]:
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            total_people += 1
            bbox_xyxy = yolo_bbox_to_xyxy(entry["bbox"])
            if bbox_xyxy is None or not mpii_people:
                unmatched_people += 1
                removed_keypoints += drop_keypoints_outside_bbox(entry["kpts"], bbox_xyxy)
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            match_idx, _, _, _ = match_mpii_person(bbox_xyxy, mpii_people, available, match_threshold)
            if match_idx is None:
                unmatched_people += 1
                removed_keypoints += drop_keypoints_outside_bbox(entry["kpts"], bbox_xyxy)
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            matched_people += 1
            available.discard(match_idx)
            mpii_entry = mpii_people[match_idx]
            replaced_keypoints += replace_keypoints(entry["kpts"], mpii_entry["kpts"], replaced_counts)
            removed_keypoints += drop_keypoints_outside_bbox(entry["kpts"], bbox_xyxy)
            output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))

        with (output_dir / label_path.name).open("w", encoding="utf-8") as handle:
            handle.write("\n".join(output_lines))

        render_progress(index, total_files, start_time, label_path.name)

    if total_files:
        print()

    print("Finished COCO keypoint correction.")
    print(f"Files processed: {total_files}")
    print(f"Files missing MPII labels: {missing_mpii_files}")
    print(f"Total persons: {total_people}")
    print(f"Persons matched: {matched_people}")
    print(f"Persons unmatched: {unmatched_people}")
    print(f"Keypoints replaced: {replaced_keypoints}")
    print(f"Keypoints removed outside bbox: {removed_keypoints}")
    print("Replacement counts by keypoint:")
    for name in COCO_KPT_NAMES:
        if name in replaced_counts:
            print(f"- {name}: {replaced_counts[name]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct COCO pose keypoints using MPII annotations."
    )
    parser.add_argument(
        "--labels-x",
        type=Path,
        default=Path("labels-x"),
        help="Folder with YOLO pose labels to correct (default: labels-x).",
    )
    parser.add_argument(
        "--labels-mpii",
        type=Path,
        default=Path("labels-mpii"),
        help="Folder with MPII pose labels (default: labels-mpii).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labels"),
        help="Output folder for merged labels (default: labels).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        help="Minimum ratio of MPII keypoints inside bbox to match (default: 0.5).",
    )

    args = parser.parse_args()
    if not (0.0 <= args.match_threshold <= 1.0):
        raise ValueError("--match-threshold must be between 0 and 1.")

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    labels_x = resolve_path(args.labels_x, cwd)
    labels_mpii = resolve_path(args.labels_mpii, cwd)
    output_dir = resolve_path(args.output, cwd)

    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")

    process_files(labels_x, labels_mpii, output_dir, args.match_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
