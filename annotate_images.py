from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional
import warnings

from PIL import Image, ImageOps
from ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
MATCH_KPT_WEIGHT = 0.6
MATCH_CENTER_WEIGHT = 0.3
MATCH_IOU_WEIGHT = 0.1
MATCH_MAX_COST = 0.55
MATCH_UNMATCHED_COST = 0.65
NORMALIZED_DIAGONAL = math.sqrt(2)


def resolve_model_path(model_url: str, script_dir: Path) -> str:
    if "://" in model_url:
        return model_url
    model_path = Path(model_url)
    if model_path.is_absolute():
        return str(model_path)
    return str(script_dir / model_path)


def format_float(value: float) -> str:
    return f"{value:.6f}"


@dataclass
class PoseData:
    boxes_xywh: list[list[float]]
    classes: list[int]
    boxes_conf: Optional[list[float]]
    keypoints_xy: Optional[list[list[list[float]]]]
    keypoints_conf: Optional[list[list[float]]]


def write_pose_labels(output_path: Path, pose: Optional[PoseData], kp_conf: float) -> None:
    lines: list[str] = []
    if pose is None or not pose.boxes_xywh:
        output_path.write_text("", encoding="utf-8")
        return

    boxes = pose.boxes_xywh
    classes = pose.classes
    keypoints_xy = pose.keypoints_xy
    keypoints_conf = pose.keypoints_conf

    for idx in range(len(boxes)):
        cls = int(classes[idx]) if classes else 0
        xywh = boxes[idx]
        parts: list[str] = [str(cls)] + [format_float(v) for v in xywh]

        if keypoints_xy is not None and len(keypoints_xy) > idx:
            kpt_xy = keypoints_xy[idx]
            kpt_conf = keypoints_conf[idx] if keypoints_conf is not None else None
            for kp_index, (x, y) in enumerate(kpt_xy):
                conf_value = float(kpt_conf[kp_index]) if kpt_conf is not None else 1.0
                is_visible = conf_value >= kp_conf
                if not is_visible:
                    x, y = 0.0, 0.0
                parts.extend(
                    [
                        format_float(x),
                        format_float(y),
                        "2" if is_visible else "0",
                    ]
                )

        lines.append(" ".join(parts))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def extract_pose(result) -> Optional[PoseData]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    box_xywh = boxes.xywhn.tolist()
    classes = boxes.cls.tolist() if boxes.cls is not None else [0] * len(box_xywh)
    box_conf = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else None

    keypoints = result.keypoints
    if keypoints is None or len(keypoints) == 0:
        kpt_xy = None
        kpt_conf = None
    else:
        kpt_xy = keypoints.xyn.tolist()
        kpt_conf = keypoints.conf.tolist() if getattr(keypoints, "conf", None) is not None else None

    return PoseData(
        boxes_xywh=box_xywh,
        classes=[int(v) for v in classes],
        boxes_conf=box_conf,
        keypoints_xy=kpt_xy,
        keypoints_conf=kpt_conf,
    )


def align_flipped_pose(pose: PoseData, flip_idx: list[int]) -> PoseData:        
    flipped_boxes = []
    for x, y, w, h in pose.boxes_xywh:
        flipped_boxes.append([1.0 - x, y, w, h])

    flipped_kpt_xy = None
    flipped_kpt_conf = None
    expected_kpt = len(flip_idx)
    warned = False
    if pose.keypoints_xy is not None:
        flipped_kpt_xy = []
        for person in pose.keypoints_xy:
            mirrored = []
            use_mapping = len(person) == expected_kpt
            if not use_mapping and not warned:
                warnings.warn(
                    "Expected 17 keypoints for flip mapping; mirroring without swapping.",
                    RuntimeWarning,
                )
                warned = True
            if use_mapping:
                for idx in range(expected_kpt):
                    src_idx = flip_idx[idx]
                    x, y = person[src_idx]
                    mirrored.append([1.0 - x, y])
            else:
                for x, y in person:
                    mirrored.append([1.0 - x, y])
            flipped_kpt_xy.append(mirrored)

        if pose.keypoints_conf is not None:
            flipped_kpt_conf = []
            for person_conf in pose.keypoints_conf:
                use_mapping = len(person_conf) == expected_kpt
                if not use_mapping and not warned:
                    warnings.warn(
                        "Expected 17 keypoint confidences for flip mapping; mirroring without swapping.",
                        RuntimeWarning,
                    )
                    warned = True
                if use_mapping:
                    mirrored_conf = [
                        float(person_conf[flip_idx[idx]]) for idx in range(expected_kpt)
                    ]
                else:
                    mirrored_conf = [float(v) for v in person_conf]
                flipped_kpt_conf.append(mirrored_conf)

    return PoseData(
        boxes_xywh=flipped_boxes,
        classes=pose.classes,
        boxes_conf=pose.boxes_conf,
        keypoints_xy=flipped_kpt_xy,
        keypoints_conf=flipped_kpt_conf,
    )


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    half_w = w / 2.0
    half_h = h / 2.0
    return x - half_w, y - half_h, x + half_w, y + half_h


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def keypoint_distance(
    a_xy: list[list[float]],
    b_xy: list[list[float]],
    a_conf: Optional[list[float]],
    b_conf: Optional[list[float]],
) -> Optional[float]:
    total = 0.0
    weight_sum = 0.0
    count = min(len(a_xy), len(b_xy))
    for idx in range(count):
        if a_conf is not None and b_conf is not None:
            weight = min(float(a_conf[idx]), float(b_conf[idx]))
        elif a_conf is not None:
            weight = float(a_conf[idx])
        elif b_conf is not None:
            weight = float(b_conf[idx])
        else:
            weight = 1.0
        if weight <= 0:
            continue
        dx = a_xy[idx][0] - b_xy[idx][0]
        dy = a_xy[idx][1] - b_xy[idx][1]
        total += math.hypot(dx, dy) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return (total / weight_sum) / NORMALIZED_DIAGONAL


def pair_cost(normal: PoseData, flipped: PoseData, n_idx: int, f_idx: int) -> float:
    n_box = normal.boxes_xywh[n_idx]
    f_box = flipped.boxes_xywh[f_idx]
    center_dist = math.hypot(n_box[0] - f_box[0], n_box[1] - f_box[1]) / NORMALIZED_DIAGONAL
    iou_cost = 1.0 - box_iou(n_box, f_box)

    kpt_cost = None
    if normal.keypoints_xy is not None and flipped.keypoints_xy is not None:
        n_kpt = normal.keypoints_xy[n_idx]
        f_kpt = flipped.keypoints_xy[f_idx]
        n_conf = normal.keypoints_conf[n_idx] if normal.keypoints_conf is not None else None
        f_conf = flipped.keypoints_conf[f_idx] if flipped.keypoints_conf is not None else None
        kpt_cost = keypoint_distance(n_kpt, f_kpt, n_conf, f_conf)

    if kpt_cost is None:
        total_weight = MATCH_CENTER_WEIGHT + MATCH_IOU_WEIGHT
        center_weight = MATCH_CENTER_WEIGHT / total_weight
        iou_weight = MATCH_IOU_WEIGHT / total_weight
        return center_weight * center_dist + iou_weight * iou_cost

    return (
        MATCH_KPT_WEIGHT * kpt_cost
        + MATCH_CENTER_WEIGHT * center_dist
        + MATCH_IOU_WEIGHT * iou_cost
    )


def solve_assignment(cost_matrix: list[list[float]], unmatched_cost: float) -> list[int]:
    if not cost_matrix:
        return []
    n = len(cost_matrix)
    m = len(cost_matrix[0])
    padded = [row + [unmatched_cost] * n for row in cost_matrix]
    m_padded = m + n
    u = [0.0] * (n + 1)
    v = [0.0] * (m_padded + 1)
    p = [0] * (m_padded + 1)
    way = [0] * (m_padded + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m_padded + 1)
        used = [False] * (m_padded + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m_padded + 1):
                if not used[j]:
                    cur = padded[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m_padded + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m_padded + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def match_pose_pairs(
    normal: PoseData, flipped: PoseData
) -> tuple[dict[int, int], list[int], list[int]]:
    n = len(normal.boxes_xywh)
    m = len(flipped.boxes_xywh)
    if n == 0 or m == 0:
        return {}, list(range(n)), list(range(m))

    cost_matrix = [
        [pair_cost(normal, flipped, i, j) for j in range(m)] for i in range(n)
    ]
    assignment = solve_assignment(cost_matrix, MATCH_UNMATCHED_COST)
    normal_to_flipped: dict[int, int] = {}
    matched_flipped: set[int] = set()
    for i, j in enumerate(assignment):
        if j < 0 or j >= m:
            continue
        if cost_matrix[i][j] <= MATCH_MAX_COST:
            normal_to_flipped[i] = j
            matched_flipped.add(j)

    unmatched_normal = [i for i in range(n) if i not in normal_to_flipped]
    unmatched_flipped = [j for j in range(m) if j not in matched_flipped]
    return normal_to_flipped, unmatched_normal, unmatched_flipped


def merge_pose(normal: Optional[PoseData], flipped: Optional[PoseData]) -> Optional[PoseData]:
    if normal is None and flipped is None:
        return None
    if normal is None:
        return flipped
    if flipped is None:
        return normal

    normal_to_flipped, _, unmatched_flipped = match_pose_pairs(normal, flipped)
    merged_boxes: list[list[float]] = []
    merged_classes: list[int] = []
    merged_box_conf: Optional[list[float]] = [] if normal.boxes_conf or flipped.boxes_conf else None
    merged_kpt_xy: Optional[list[list[list[float]]]] = [] if normal.keypoints_xy or flipped.keypoints_xy else None
    merged_kpt_conf: Optional[list[list[float]]] = [] if normal.keypoints_conf or flipped.keypoints_conf else None

    def append_pose(pose: PoseData, idx: int) -> None:
        merged_boxes.append(pose.boxes_xywh[idx])
        merged_classes.append(pose.classes[idx] if pose.classes else 0)
        if merged_box_conf is not None:
            if pose.boxes_conf is None:
                merged_box_conf.append(1.0)
            else:
                merged_box_conf.append(float(pose.boxes_conf[idx]))
        if merged_kpt_xy is not None:
            if pose.keypoints_xy is None:
                merged_kpt_xy.append([])
                if merged_kpt_conf is not None:
                    merged_kpt_conf.append([])
            else:
                merged_kpt_xy.append(pose.keypoints_xy[idx])
                if merged_kpt_conf is not None:
                    if pose.keypoints_conf is None:
                        merged_kpt_conf.append([1.0] * len(pose.keypoints_xy[idx]))
                    else:
                        merged_kpt_conf.append([float(v) for v in pose.keypoints_conf[idx]])

    for n_idx in range(len(normal.boxes_xywh)):
        f_idx = normal_to_flipped.get(n_idx)
        if f_idx is None:
            append_pose(normal, n_idx)
            continue

        n_conf = float(normal.boxes_conf[n_idx]) if normal.boxes_conf is not None else 1.0
        f_conf = float(flipped.boxes_conf[f_idx]) if flipped.boxes_conf is not None else 1.0
        weight_sum = n_conf + f_conf if n_conf + f_conf > 0 else 1.0
        n_box = normal.boxes_xywh[n_idx]
        f_box = flipped.boxes_xywh[f_idx]
        merged_boxes.append(
            [
                (n_box[0] * n_conf + f_box[0] * f_conf) / weight_sum,
                (n_box[1] * n_conf + f_box[1] * f_conf) / weight_sum,
                (n_box[2] * n_conf + f_box[2] * f_conf) / weight_sum,
                (n_box[3] * n_conf + f_box[3] * f_conf) / weight_sum,
            ]
        )
        merged_classes.append(normal.classes[n_idx] if normal.classes else 0)
        if merged_box_conf is not None:
            merged_box_conf.append(max(n_conf, f_conf))

        if merged_kpt_xy is not None:
            n_kpt = normal.keypoints_xy[n_idx] if normal.keypoints_xy is not None else None
            f_kpt = flipped.keypoints_xy[f_idx] if flipped.keypoints_xy is not None else None
            if n_kpt is None and f_kpt is None:
                merged_kpt_xy.append([])
                if merged_kpt_conf is not None:
                    merged_kpt_conf.append([])
            elif n_kpt is None:
                merged_kpt_xy.append(f_kpt)
                if merged_kpt_conf is not None:
                    if flipped.keypoints_conf is None:
                        merged_kpt_conf.append([1.0] * len(f_kpt))
                    else:
                        merged_kpt_conf.append([float(v) for v in flipped.keypoints_conf[f_idx]])
            elif f_kpt is None:
                merged_kpt_xy.append(n_kpt)
                if merged_kpt_conf is not None:
                    if normal.keypoints_conf is None:
                        merged_kpt_conf.append([1.0] * len(n_kpt))
                    else:
                        merged_kpt_conf.append([float(v) for v in normal.keypoints_conf[n_idx]])
            else:
                merged_person_xy = []
                merged_person_conf = []
                kp_len = max(len(n_kpt), len(f_kpt))
                for kp_idx in range(kp_len):
                    if kp_idx >= len(n_kpt):
                        merged_person_xy.append([f_kpt[kp_idx][0], f_kpt[kp_idx][1]])
                        if merged_kpt_conf is not None:
                            if flipped.keypoints_conf is None:
                                merged_person_conf.append(1.0)
                            else:
                                merged_person_conf.append(float(flipped.keypoints_conf[f_idx][kp_idx]))
                        continue
                    if kp_idx >= len(f_kpt):
                        merged_person_xy.append([n_kpt[kp_idx][0], n_kpt[kp_idx][1]])
                        if merged_kpt_conf is not None:
                            if normal.keypoints_conf is None:
                                merged_person_conf.append(1.0)
                            else:
                                merged_person_conf.append(float(normal.keypoints_conf[n_idx][kp_idx]))
                        continue

                    n_kp_conf = (
                        float(normal.keypoints_conf[n_idx][kp_idx])
                        if normal.keypoints_conf is not None
                        else 1.0
                    )
                    f_kp_conf = (
                        float(flipped.keypoints_conf[f_idx][kp_idx])
                        if flipped.keypoints_conf is not None
                        else 1.0
                    )
                    kp_weight_sum = n_kp_conf + f_kp_conf
                    if kp_weight_sum > 0:
                        merged_x = (n_kpt[kp_idx][0] * n_kp_conf + f_kpt[kp_idx][0] * f_kp_conf) / kp_weight_sum
                        merged_y = (n_kpt[kp_idx][1] * n_kp_conf + f_kpt[kp_idx][1] * f_kp_conf) / kp_weight_sum
                    else:
                        merged_x = 0.0
                        merged_y = 0.0
                    merged_person_xy.append([merged_x, merged_y])
                    if merged_kpt_conf is not None:
                        merged_person_conf.append(max(n_kp_conf, f_kp_conf))
                merged_kpt_xy.append(merged_person_xy)
                if merged_kpt_conf is not None:
                    merged_kpt_conf.append(merged_person_conf)

    for f_idx in unmatched_flipped:
        append_pose(flipped, f_idx)

    return PoseData(
        boxes_xywh=merged_boxes,
        classes=merged_classes,
        boxes_conf=merged_box_conf if merged_box_conf is not None else None,
        keypoints_xy=merged_kpt_xy,
        keypoints_conf=merged_kpt_conf,
    )


def annotate_images(
    root: Path,
    model_url: str = "yolo11x-pose.pt",
    img_size: int = 640,
    conf: float = 0.3,
    kp_conf: float = 0.4,
    labels_dir_name: str = "labels",
    script_dir: Optional[Path] = None,
) -> None:
    header_lines = [
        "Annotating images with normal + flipped inference.",
        f"- Data root (CWD): {root}",
    ]
    if script_dir is not None:
        header_lines.append(f"- Script dir: {script_dir}")
    header_lines.append(
        f"- model={model_url}, img_size={img_size}, conf={conf}, kp_conf={kp_conf}"
    )
    header_lines.append(f"- labels={labels_dir_name}")
    print("\n".join(header_lines))
    images_dir = root / "images"
    labels_dir = root / labels_dir_name
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    labels_dir.mkdir(exist_ok=True)

    model = YOLO(model_url)

    image_paths = [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    iterator = image_paths
    if tqdm is not None:
        iterator = tqdm(image_paths, desc="Annotating", unit="image")

    for image_path in iterator:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            flipped = ImageOps.mirror(image)

            normal_results = model.predict(
                source=image,
                save=False,
                verbose=False,
                imgsz=img_size,
                conf=conf,
            )
            flipped_results = model.predict(
                source=flipped,
                save=False,
                verbose=False,
                imgsz=img_size,
                conf=conf,
            )

        normal_pose = extract_pose(normal_results[0]) if normal_results else None
        flipped_pose = extract_pose(flipped_results[0]) if flipped_results else None
        if flipped_pose is not None:
            flipped_pose = align_flipped_pose(flipped_pose, FLIP_IDX)
        merged_pose = merge_pose(normal_pose, flipped_pose)
        output_path = labels_dir / f"{image_path.stem}.txt"
        write_pose_labels(output_path, merged_pose, kp_conf)

    print(f"Annotated {len(image_paths)} images. Labels saved to {labels_dir}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate images with pose labels.")
    parser.add_argument(
        "--model",
        dest="model_url",
        default="yolo11x-pose.pt",
        help="YOLO pose model path or URL.",
    )
    parser.add_argument(
        "--img-size",
        dest="img_size",
        type=int,
        default=640,
        help="Model inference image size.",
    )
    parser.add_argument(
        "--conf",
        dest="conf",
        type=float,
        default=0.3,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--kp-conf",
        dest="kp_conf",
        type=float,
        default=0.4,
        help="Keypoint confidence threshold for visibility and masking.",       
    )
    parser.add_argument(
        "--labels",
        dest="labels_dir_name",
        default="labels",
        help="Labels output folder name inside the root directory.",
    )
    args = parser.parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    model_url = resolve_model_path(args.model_url, script_dir)
    annotate_images(
        cwd,
        model_url=model_url,
        img_size=args.img_size,
        conf=args.conf,
        kp_conf=args.kp_conf,
        labels_dir_name=args.labels_dir_name,
        script_dir=script_dir,
    )
