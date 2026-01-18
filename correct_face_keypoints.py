from __future__ import annotations

import argparse
import math
import time
from pathlib import Path


FACE_INDICES = [0, 1, 2, 3, 4]
FACE_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
DEFAULT_IOU_THRESHOLD = 0.2
DEFAULT_CENTER_DIST_THRESHOLD = 0.05


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


def face_bbox(kpts: list[list[float]]) -> tuple[float, float, float, float] | None:
    pts = []
    for idx in FACE_INDICES:
        if idx < len(kpts) and is_valid_kpt(kpts[idx]):
            pts.append((kpts[idx][0], kpts[idx][1]))
    if not pts:
        return None
    xs, ys = zip(*pts)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x <= min_x or max_y <= min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def bbox_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def match_face(
    orig: dict,
    faces: list[dict],
    available: set[int],
    allow_center_match: bool,
    iou_threshold: float,
    center_dist_threshold: float,
) -> int | None:
    orig_bbox = face_bbox(orig["kpts"])
    if orig_bbox is None:
        return None

    best_iou = -1.0
    best_idx = None
    for idx in available:
        face_bbox_val = face_bbox(faces[idx]["kpts"])
        if face_bbox_val is None:
            continue
        iou = bbox_iou(orig_bbox, face_bbox_val)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_iou >= iou_threshold and best_idx is not None:
        return best_idx

    if not allow_center_match:
        return None

    orig_center = bbox_center(orig_bbox)
    best_dist = None
    best_dist_idx = None
    for idx in available:
        face_bbox_val = face_bbox(faces[idx]["kpts"])
        if face_bbox_val is None:
            continue
        face_center = bbox_center(face_bbox_val)
        dist = math.hypot(orig_center[0] - face_center[0], orig_center[1] - face_center[1])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_dist_idx = idx

    if best_dist is not None and best_dist <= center_dist_threshold:
        return best_dist_idx
    return None


def merge_face_keypoints(
    orig_kpts: list[list[float]],
    face_kpts: list[list[float]],
    weight: float,
) -> list[list[float]]:
    merged = [kp[:] for kp in orig_kpts]
    for idx in FACE_INDICES:
        if idx >= len(merged):
            continue
        face_kp = face_kpts[idx] if idx < len(face_kpts) else [0.0, 0.0, 0.0]
        orig_kp = merged[idx]
        face_valid = is_valid_kpt(face_kp)
        orig_valid = is_valid_kpt(orig_kp)

        if not face_valid:
            merged[idx] = [0.0, 0.0, 0.0]
        elif face_valid and not orig_valid:
            merged[idx] = face_kp[:]
        else:
            merged[idx] = [
                orig_kp[0] * (1.0 - weight) + face_kp[0] * weight,
                orig_kp[1] * (1.0 - weight) + face_kp[1] * weight,
                face_kp[2],
            ]
    return merged


def process_files(
    labels_orig: Path,
    labels_face: Path,
    output_dir: Path,
    face_weight: float,
    match_iou_only: bool,
    match_iou_threshold: float,
    match_center_threshold: float,
) -> None:
    print("Starting face keypoint correction.")
    print(
        "Parameters:\n"
        f"- Labels orig: {labels_orig}\n"
        f"- Labels face: {labels_face}\n"
        f"- Output: {output_dir}\n"
        f"- Face weight: {face_weight}\n"
        f"- IoU threshold: {match_iou_threshold}\n"
        f"- Center dist threshold: {match_center_threshold}\n"
        f"- Match IoU only: {match_iou_only}"
    )

    if not labels_orig.is_dir():
        raise FileNotFoundError(f"labels-orig directory not found: {labels_orig}")

    if not labels_face.is_dir():
        print(f"Warning: labels-face directory not found: {labels_face}. Using originals only.")

    output_dir.mkdir(parents=True, exist_ok=True)

    orig_files = sorted(labels_orig.glob("*.txt"))
    total_files = len(orig_files)
    print(f"Found {total_files} label files in '{labels_orig}'.")

    total_persons = 0
    matched_faces = 0
    unmatched_faces = 0
    deviation_sum = [0.0 for _ in FACE_INDICES]
    deviation_count = [0 for _ in FACE_INDICES]

    start_time = time.time()
    for index, orig_path in enumerate(orig_files, start=1):
        face_path = labels_face / orig_path.name
        output_path = output_dir / orig_path.name

        orig_entries = []
        with orig_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = parse_yolo_pose(line)
                if entry is not None:
                    orig_entries.append(entry)

        face_entries = []
        if labels_face.is_dir() and face_path.exists():
            with face_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = parse_yolo_pose(line)
                    if entry is not None:
                        face_entries.append(entry)

        face_people = [entry for entry in face_entries if entry["class_id"] == 0]
        available_faces = set(range(len(face_people)))

        output_lines = []
        for entry in orig_entries:
            if entry["class_id"] != 0 or not entry["kpts"]:
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            total_persons += 1
            match_idx = None
            if face_people:
                match_idx = match_face(
                    entry,
                    face_people,
                    available_faces,
                    not match_iou_only,
                    match_iou_threshold,
                    match_center_threshold,
                )
            if match_idx is None:
                unmatched_faces += 1
                output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], entry["kpts"]))
                continue

            matched_faces += 1
            available_faces.discard(match_idx)
            face_entry = face_people[match_idx]

            for idx, kp_idx in enumerate(FACE_INDICES):
                if kp_idx >= len(entry["kpts"]) or kp_idx >= len(face_entry["kpts"]):
                    continue
                orig_kp = entry["kpts"][kp_idx]
                face_kp = face_entry["kpts"][kp_idx]
                if is_valid_kpt(orig_kp) and is_valid_kpt(face_kp):
                    deviation_sum[idx] += math.hypot(orig_kp[0] - face_kp[0], orig_kp[1] - face_kp[1])
                    deviation_count[idx] += 1

            merged_kpts = merge_face_keypoints(entry["kpts"], face_entry["kpts"], face_weight)
            output_lines.append(format_yolo_pose(entry["class_id"], entry["bbox"], merged_kpts))

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(output_lines))

        render_progress(index, total_files, start_time, orig_path.name)

    if total_files:
        print()

    print("Finished face keypoint correction.")
    print(f"Files processed: {total_files}")
    print(f"Total persons: {total_persons}")
    print(f"Faces matched: {matched_faces}")
    print(f"Faces unmatched: {unmatched_faces}")
    print("Average deviations (orig vs face, normalized coords):")
    for name, total, count in zip(FACE_NAMES, deviation_sum, deviation_count):
        if count:
            avg = total / count
            print(f"- {name}: {avg:.6f} (n={count})")
        else:
            print(f"- {name}: n/a (n=0)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Correct YOLO pose face keypoints.")
    parser.add_argument(
        "--labels-orig",
        type=Path,
        default=Path("labels-orig"),
        help="Folder with original pose annotations (default: labels-orig).",
    )
    parser.add_argument(
        "--labels-face",
        type=Path,
        default=Path("labels-face"),
        help="Folder with face keypoint annotations (default: labels-face).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labels"),
        help="Output folder for merged labels (default: labels).",
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=1.0,
        help="Weight for face keypoints when blending (0..1, default: 1.0).",
    )
    parser.add_argument(
        "--match-iou-only",
        action="store_true",
        help="Match faces using IoU only (skip center-distance fallback).",
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for matching faces (default: 0.2).",
    )
    parser.add_argument(
        "--match-center-threshold",
        type=float,
        default=DEFAULT_CENTER_DIST_THRESHOLD,
        help="Center distance threshold for matching faces (default: 0.05).",
    )

    args = parser.parse_args()
    if not (0.0 <= args.face_weight <= 1.0):
        raise ValueError("--face-weight must be between 0 and 1.")

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    labels_orig = resolve_path(args.labels_orig, cwd)
    labels_face = resolve_path(args.labels_face, cwd)
    output_dir = resolve_path(args.output, cwd)

    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    if not (0.0 <= args.match_iou_threshold <= 1.0):
        raise ValueError("--match-iou-threshold must be between 0 and 1.")
    if args.match_center_threshold <= 0:
        raise ValueError("--match-center-threshold must be greater than 0.")

    process_files(
        labels_orig,
        labels_face,
        output_dir,
        args.face_weight,
        args.match_iou_only,
        args.match_iou_threshold,
        args.match_center_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
