from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".bmp",
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
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress(current: int, total: int, start_time: float, label: str) -> None:
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
    tail = label
    if len(tail) > 36:
        tail = f"...{tail[-33:]}"
    print(f"\r[{bar}] {current}/{total} {percent:3d}% ETA {eta} {tail}   ", end="", flush=True)


def parse_yolo_pose_line(line: str) -> dict | None:
    parts = line.strip().split()
    if not parts:
        return None

    class_id = int(float(parts[0]))
    bbox = []
    if len(parts) >= 5:
        bbox = [float(v) for v in parts[1:5]]

    kpts = []
    kpt_values = [float(v) for v in parts[5:]]
    for index in range(0, len(kpt_values), 3):
        if index + 2 >= len(kpt_values):
            break
        kpts.append([kpt_values[index], kpt_values[index + 1], kpt_values[index + 2]])

    return {"class_id": class_id, "bbox": bbox, "kpts": kpts}


def format_yolo_pose_line(class_id: int, bbox: list[float], keypoints: list[list[float]]) -> str:
    parts = [str(class_id)]
    parts.extend(f"{v:.6f}" for v in bbox)
    for point in keypoints:
        parts.append(f"{point[0]:.6f}")
        parts.append(f"{point[1]:.6f}")
        parts.append(f"{point[2]:.6f}")
    return " ".join(parts)


def extract_numeric_index(stem: str) -> int | None:
    match = re.search(r"(\d+)(?!.*\d)", stem)
    if not match:
        return None
    return int(match.group(1))


def collect_indexed_images(images_dir: Path) -> list[dict]:
    indexed = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index = extract_numeric_index(image_path.stem)
        if index is None:
            continue
        indexed.append({"index": index, "path": image_path})
    indexed.sort(key=lambda item: (item["index"], item["path"].name.lower()))
    return indexed


def load_pose_entries(label_path: Path, class_id: int) -> list[dict]:
    entries: list[dict] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_yolo_pose_line(line)
            if parsed is None:
                continue
            if parsed["class_id"] == class_id and len(parsed["bbox"]) == 4:
                entries.append(parsed)
    return entries


def select_target_image(indexed_images: list[dict], target_index: int) -> tuple[dict, bool]:
    candidates = [item for item in indexed_images if item["index"] == target_index]
    if candidates:
        if len(candidates) > 1:
            names = ", ".join(candidate["path"].name for candidate in candidates)
            print(
                "Warning: multiple images matched the requested index. "
                f"Using '{candidates[0]['path'].name}'. Candidates: {names}"
            )
        return candidates[0], False

    next_candidates = [item for item in indexed_images if item["index"] > target_index]
    if not next_candidates:
        raise FileNotFoundError(
            f"Target image index {target_index} was not found, and no next indexed image exists."
        )

    next_candidates.sort(key=lambda item: (item["index"], item["path"].name.lower()))
    chosen = next_candidates[0]
    return chosen, True


def select_source_image_and_objects(
    indexed_images: list[dict],
    labels_dir: Path,
    target_index: int,
    class_id: int,
    source_object: int,
) -> tuple[dict, Path, list[dict]]:
    previous_candidates = [item for item in indexed_images if item["index"] < target_index]
    previous_candidates.sort(key=lambda item: item["index"], reverse=True)

    for candidate in previous_candidates:
        label_path = labels_dir / f"{candidate['path'].stem}.txt"
        if not label_path.exists():
            continue

        pose_entries = load_pose_entries(label_path, class_id)
        if not pose_entries:
            continue

        if source_object >= 0:
            if source_object >= len(pose_entries):
                continue
            return candidate, label_path, [pose_entries[source_object]]

        return candidate, label_path, pose_entries

    if source_object >= 0:
        raise FileNotFoundError(
            "Could not find a previous indexed image with labels containing "
            f"class {class_id} object #{source_object}."
        )
    raise FileNotFoundError(
        "Could not find a previous indexed image with labels containing "
        f"class {class_id} annotations."
    )


def is_valid_source_keypoint(point: list[float]) -> bool:
    return point[2] > 0 and not (point[0] == 0.0 and point[1] == 0.0)


def predict_keypoint_with_backcheck(
    prev_gray: np.ndarray,
    target_gray: np.ndarray,
    point_norm: tuple[float, float],
    lk_params: dict,
    backcheck_threshold: float,
) -> tuple[bool, tuple[float, float] | None, str]:
    flow_height, flow_width = prev_gray.shape[:2]

    x = float(point_norm[0]) * flow_width
    y = float(point_norm[1]) * flow_height
    x = min(max(x, 0.0), flow_width - 1.0)
    y = min(max(y, 0.0), flow_height - 1.0)

    point_prev = np.array([[[x, y]]], dtype=np.float32)
    point_next, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        target_gray,
        point_prev,
        None,
        **lk_params,
    )
    if point_next is None or status_forward is None or int(status_forward[0, 0]) != 1:
        return False, None, "forward_fail"

    point_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        target_gray,
        prev_gray,
        point_next,
        None,
        **lk_params,
    )
    if point_back is None or status_backward is None or int(status_backward[0, 0]) != 1:
        return False, None, "backward_fail"

    forward_xy = point_next[0, 0]
    backward_xy = point_back[0, 0]
    back_error = float(np.linalg.norm(point_prev[0, 0] - backward_xy))
    if back_error > backcheck_threshold:
        return False, None, "backcheck_fail"

    x_next = float(forward_xy[0])
    y_next = float(forward_xy[1])
    if x_next < 0.0 or x_next >= flow_width or y_next < 0.0 or y_next >= flow_height:
        return False, None, "out_of_frame"

    return True, (x_next / flow_width, y_next / flow_height), "ok"


def predict_objects(
    source_objects: list[dict],
    prev_gray: np.ndarray,
    target_gray: np.ndarray,
    num_keypoints: int,
    lk_params: dict,
    backcheck_threshold: float,
    show_progress: bool = True,
) -> tuple[list[str], dict[str, int]]:
    output_lines: list[str] = []
    base_win_w, base_win_h = lk_params["winSize"]
    retry_win_w = max(2, int(round(base_win_w * 1.5)))
    retry_win_h = max(2, int(round(base_win_h * 1.5)))
    retry_lk_params = dict(lk_params)
    retry_lk_params["winSize"] = (retry_win_w, retry_win_h)
    retry_backcheck_threshold = backcheck_threshold * 2.0
    stats = {
        "objects_appended": 0,
        "keypoints_total": 0,
        "keypoints_source_invalid": 0,
        "keypoints_predicted": 0,
        "keypoints_retry_attempted": 0,
        "keypoints_retry_predicted": 0,
        "keypoints_kept_previous": 0,
        "forward_fail": 0,
        "backward_fail": 0,
        "backcheck_fail": 0,
        "out_of_frame": 0,
    }

    total_steps = max(1, len(source_objects) * num_keypoints)
    step_index = 0
    start_time = time.time()

    for object_idx, entry in enumerate(source_objects, start=1):
        bbox = entry["bbox"][:]
        source_kpts = entry["kpts"]
        predicted_kpts: list[list[float]] = []

        for kp_idx in range(num_keypoints):
            step_index += 1
            stats["keypoints_total"] += 1

            if kp_idx >= len(source_kpts):
                predicted_kpts.append([0.0, 0.0, 0.0])
                stats["keypoints_source_invalid"] += 1
                if show_progress:
                    render_progress(step_index, total_steps, start_time, f"obj{object_idx} kp{kp_idx}")
                continue

            source_point = source_kpts[kp_idx]
            if not is_valid_source_keypoint(source_point):
                predicted_kpts.append([0.0, 0.0, 0.0])
                stats["keypoints_source_invalid"] += 1
                if show_progress:
                    render_progress(step_index, total_steps, start_time, f"obj{object_idx} kp{kp_idx}")
                continue

            ok, predicted_xy, reason = predict_keypoint_with_backcheck(
                prev_gray,
                target_gray,
                (source_point[0], source_point[1]),
                lk_params,
                backcheck_threshold,
            )
            if not ok or predicted_xy is None:
                stats["keypoints_retry_attempted"] += 1
                ok_retry, predicted_xy_retry, reason_retry = predict_keypoint_with_backcheck(
                    prev_gray,
                    target_gray,
                    (source_point[0], source_point[1]),
                    retry_lk_params,
                    retry_backcheck_threshold,
                )
                if not ok_retry or predicted_xy_retry is None:
                    predicted_kpts.append([source_point[0], source_point[1], source_point[2]])
                    stats["keypoints_kept_previous"] += 1
                    stats[reason_retry] += 1
                    if show_progress:
                        render_progress(step_index, total_steps, start_time, f"obj{object_idx} kp{kp_idx}")
                    continue

                predicted_kpts.append([predicted_xy_retry[0], predicted_xy_retry[1], source_point[2]])
                stats["keypoints_predicted"] += 1
                stats["keypoints_retry_predicted"] += 1
                if show_progress:
                    render_progress(step_index, total_steps, start_time, f"obj{object_idx} kp{kp_idx}")
                continue

            predicted_kpts.append([predicted_xy[0], predicted_xy[1], source_point[2]])
            stats["keypoints_predicted"] += 1
            if show_progress:
                render_progress(step_index, total_steps, start_time, f"obj{object_idx} kp{kp_idx}")

        output_lines.append(format_yolo_pose_line(entry["class_id"], bbox, predicted_kpts))
        stats["objects_appended"] += 1

    if show_progress and step_index:
        print()

    return output_lines, stats


def run_prediction(
    images_dir: Path,
    labels_dir: Path,
    target_index_start: int,
    target_index_end: int | None,
    source_object: int,
    class_id: int,
    num_keypoints: int,
    backcheck_threshold: float,
    window_size: int,
) -> None:
    base_window = (window_size, window_size)
    retry_window = (
        max(2, int(round(base_window[0] * 1.5))),
        max(2, int(round(base_window[1] * 1.5))),
    )
    print(
        "Starting YOLO pose keypoint prediction from previous indexed frame using optical flow.\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Target index start: {target_index_start}\n"
        f"- Target index end: {target_index_end if target_index_end is not None else target_index_start}\n"
        f"- Source object index: {source_object if source_object >= 0 else 'all'}\n"
        f"- Class id: {class_id}\n"
        f"- Num keypoints: {num_keypoints}\n"
        "- Optical flow: Lucas-Kanade pyramidal\n"
        "- LK levels: 3\n"
        f"- Window size: {base_window[0]}x{base_window[1]}\n"
        f"- Back-check threshold (px): {backcheck_threshold:.2f}\n"
        "- Retry on flow failure: yes\n"
        f"- Retry window size: {retry_window[0]}x{retry_window[1]} (+50%)\n"
        f"- Retry back-check threshold (px): {backcheck_threshold * 2.0:.2f}\n"
        "- If retry fails: keep previous keypoint position"
    )

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    indexed_images = collect_indexed_images(images_dir)
    if not indexed_images:
        raise FileNotFoundError(f"No indexed images found in: {images_dir}")

    if target_index_end is None:
        target_image, used_next_image = select_target_image(indexed_images, target_index_start)
        if used_next_image:
            print(
                "Requested target index was missing. "
                f"Using next available image index {target_image['index']}: {target_image['path'].name}"
            )
        target_images = [target_image]
    else:
        max_available_index = max(item["index"] for item in indexed_images)
        if target_index_start > max_available_index:
            raise FileNotFoundError(
                f"Target start index {target_index_start} is greater than last available image index {max_available_index}."
            )
        effective_end = min(target_index_end, max_available_index)
        if effective_end < target_index_end:
            print(
                "Requested target end index exceeds available images. "
                f"Using last available index {effective_end}."
            )
        target_images = [
            item
            for item in indexed_images
            if target_index_start <= item["index"] <= effective_end
        ]
        if not target_images:
            raise FileNotFoundError(
                f"No images found in requested index range {target_index_start}..{effective_end}."
            )

    print(f"Target images to process: {len(target_images)}")
    is_batch_mode = target_index_end is not None
    if is_batch_mode:
        print("Batch mode: compact output (single progress bar + final stats).")

    lk_params = {
        "winSize": base_window,
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }

    batch_stats = {
        "objects_appended": 0,
        "keypoints_total": 0,
        "keypoints_predicted": 0,
        "keypoints_source_invalid": 0,
        "keypoints_retry_attempted": 0,
        "keypoints_retry_predicted": 0,
        "keypoints_kept_previous": 0,
        "forward_fail": 0,
        "backward_fail": 0,
        "backcheck_fail": 0,
        "out_of_frame": 0,
    }
    processed_targets = 0
    skipped_targets = 0
    batch_start_time = time.time()
    skipped_missing_source = 0
    skipped_runtime_error = 0

    for target_pos, target_image in enumerate(target_images, start=1):
        if not is_batch_mode:
            print(
                f"\nTarget {target_pos}/{len(target_images)}: "
                f"{target_image['path'].name} (index {target_image['index']})"
            )
        try:
            source_image, source_label_path, source_objects = select_source_image_and_objects(
                indexed_images,
                labels_dir,
                target_image["index"],
                class_id,
                source_object,
            )

            if not is_batch_mode:
                print(
                    "Source image: "
                    f"{source_image['path'].name} (index {source_image['index']}) with labels {source_label_path.name}"
                )
                print(f"Source objects selected: {len(source_objects)}")

            prev_gray = cv2.imread(str(source_image["path"]), cv2.IMREAD_GRAYSCALE)
            target_gray_raw = cv2.imread(str(target_image["path"]), cv2.IMREAD_GRAYSCALE)
            if prev_gray is None:
                raise RuntimeError(f"Failed to read source image: {source_image['path']}")
            if target_gray_raw is None:
                raise RuntimeError(f"Failed to read target image: {target_image['path']}")

            if prev_gray.shape != target_gray_raw.shape:
                target_gray = cv2.resize(
                    target_gray_raw,
                    (prev_gray.shape[1], prev_gray.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                target_gray = target_gray_raw

            predicted_lines, stats = predict_objects(
                source_objects,
                prev_gray,
                target_gray,
                num_keypoints,
                lk_params,
                backcheck_threshold,
                show_progress=not is_batch_mode,
            )

            target_label_path = labels_dir / f"{target_image['path'].stem}.txt"
            existing_lines: list[str] = []
            if target_label_path.exists():
                with target_label_path.open("r", encoding="utf-8") as handle:
                    existing_lines = [line.strip() for line in handle if line.strip()]

            merged_lines = existing_lines + predicted_lines
            with target_label_path.open("w", encoding="utf-8") as handle:
                handle.write("\n".join(merged_lines))
                if merged_lines:
                    handle.write("\n")

            processed_targets += 1
            for key in batch_stats:
                batch_stats[key] += stats[key]

            if not is_batch_mode:
                print(f"Output label file: {target_label_path}")
                print(f"Existing objects kept: {len(existing_lines)}")
                print(f"Objects appended: {stats['objects_appended']}")
                print(f"Keypoints processed: {stats['keypoints_total']}")
                print(f"Keypoints predicted: {stats['keypoints_predicted']}")
                print(f"Keypoints invalid in source: {stats['keypoints_source_invalid']}")
                print(f"Keypoints retry attempted: {stats['keypoints_retry_attempted']}")
                print(f"Keypoints predicted on retry: {stats['keypoints_retry_predicted']}")
                print(f"Keypoints kept at previous position: {stats['keypoints_kept_previous']}")
                print(
                    "Occlusion reasons: "
                    f"forward_fail={stats['forward_fail']}, "
                    f"backward_fail={stats['backward_fail']}, "
                    f"backcheck_fail={stats['backcheck_fail']}, "
                    f"out_of_frame={stats['out_of_frame']}"
                )
        except (FileNotFoundError, RuntimeError) as target_error:
            skipped_targets += 1
            if isinstance(target_error, FileNotFoundError):
                skipped_missing_source += 1
            else:
                skipped_runtime_error += 1
            if not is_batch_mode:
                print(f"Warning: skipping target {target_image['path'].name}: {target_error}")
        finally:
            if is_batch_mode:
                render_progress(
                    target_pos,
                    len(target_images),
                    batch_start_time,
                    f"index {target_image['index']}",
                )
            else:
                render_progress(target_pos, len(target_images), batch_start_time, target_image["path"].name)
                print()

    if is_batch_mode and target_images:
        print()

    if processed_targets == 0:
        raise RuntimeError("No target images were processed successfully.")

    print("\nFinished prediction batch.")
    print(f"Targets requested: {len(target_images)}")
    print(f"Targets processed: {processed_targets}")
    print(f"Targets skipped: {skipped_targets}")
    print(f"Targets skipped (missing source/labels): {skipped_missing_source}")
    print(f"Targets skipped (runtime/image read issues): {skipped_runtime_error}")
    print(f"Objects appended: {batch_stats['objects_appended']}")
    print(f"Keypoints processed: {batch_stats['keypoints_total']}")
    print(f"Keypoints predicted: {batch_stats['keypoints_predicted']}")
    print(f"Keypoints invalid in source: {batch_stats['keypoints_source_invalid']}")
    print(f"Keypoints retry attempted: {batch_stats['keypoints_retry_attempted']}")
    print(f"Keypoints predicted on retry: {batch_stats['keypoints_retry_predicted']}")
    print(f"Keypoints kept at previous position: {batch_stats['keypoints_kept_previous']}")
    print(
        "Occlusion reasons: "
        f"forward_fail={batch_stats['forward_fail']}, "
        f"backward_fail={batch_stats['backward_fail']}, "
        f"backcheck_fail={batch_stats['backcheck_fail']}, "
        f"out_of_frame={batch_stats['out_of_frame']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Predict YOLO pose keypoints for a target image index using the nearest "
            "previous indexed image with labels and Lucas-Kanade optical flow."
        )
    )
    parser.add_argument(
        "--target-index",
        type=int,
        nargs="+",
        required=True,
        help="Target index N, or target range START END.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images"),
        help="Folder with indexed images (default: images).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels"),
        help="Folder with YOLO pose labels (default: labels).",
    )
    parser.add_argument(
        "--source-object",
        type=int,
        default=0,
        help=(
            "Object index from previous label file (0-based) to propagate. "
            "Use -1 to propagate all class-id objects. Default: 0"
        ),
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class id to propagate (default: 0).",
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=17,
        help="Number of YOLO pose keypoints to write (default: 17).",
    )
    parser.add_argument(
        "--backcheck-threshold",
        type=float,
        default=1.5,
        help="Max backward-forward mismatch in pixels (default: 1.5).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="LK optical-flow window size in pixels (square). Default: 32",
    )

    args = parser.parse_args()

    try:
        if len(args.target_index) not in (1, 2):
            raise ValueError("--target-index accepts one value (N) or two values (START END).")
        target_index_start = args.target_index[0]
        target_index_end = args.target_index[1] if len(args.target_index) == 2 else None
        if target_index_end is not None and target_index_end < target_index_start:
            raise ValueError("When passing a range, END must be >= START.")

        if args.num_keypoints <= 0:
            raise ValueError("--num-keypoints must be > 0")
        if args.backcheck_threshold < 0:
            raise ValueError("--backcheck-threshold must be >= 0")
        if args.window_size <= 0:
            raise ValueError("--window-size must be > 0")

        cwd = Path.cwd()
        script_dir = Path(__file__).resolve().parent
        images_dir = resolve_path(args.images, cwd)
        labels_dir = resolve_path(args.labels, cwd)

        print(f"Data root (CWD): {cwd}")
        print(f"Script dir: {script_dir}")

        run_prediction(
            images_dir,
            labels_dir,
            target_index_start,
            target_index_end,
            args.source_object,
            args.class_id,
            args.num_keypoints,
            args.backcheck_threshold,
            args.window_size,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        sys.stdout.flush()
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
