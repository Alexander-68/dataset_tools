from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor


def format_eta(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return 'ETA --:--'
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f'ETA {minutes:02d}:{secs:02d}'


def render_progress(current: int, total: int, start_time: float, label: str) -> None:
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = '#' * filled + '-' * (bar_width - filled)
    percent = int(100 * current / total) if total > 0 else 0
    elapsed = time.time() - start_time
    eta = (elapsed / current) * (total - current) if current > 0 else None
    tail = label if len(label) <= 36 else f'...{label[-33:]}'
    sys.stdout.write(
        f'\r[{bar}] {current}/{total} {percent:3d}% {format_eta(eta)} {tail}   '
    )
    sys.stdout.flush()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def align_imgsz(imgsz: int, stride: int) -> int:
    if imgsz <= 0:
        raise ValueError('--imgsz must be > 0')
    if stride <= 0:
        raise ValueError('--stride-align must be > 0')
    return int(math.ceil(imgsz / stride) * stride)


def box_iou_xywhn(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_x1 = ax - aw / 2.0
    a_y1 = ay - ah / 2.0
    a_x2 = ax + aw / 2.0
    a_y2 = ay + ah / 2.0

    b_x1 = bx - bw / 2.0
    b_y1 = by - bh / 2.0
    b_x2 = bx + bw / 2.0
    b_y2 = by + bh / 2.0

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0

    return inter_area / denom


def nms_same_class(
    records: list[tuple[int, float, float, float, float, float]]
) -> list[tuple[int, float, float, float, float, float]]:
    if len(records) <= 1:
        return records

    by_class: dict[int, list[tuple[int, float, float, float, float, float]]] = {}
    for record in records:
        by_class.setdefault(record[0], []).append(record)

    kept: list[tuple[int, float, float, float, float, float]] = []
    for class_id in sorted(by_class):
        candidates = sorted(by_class[class_id], key=lambda item: item[5], reverse=True)
        class_kept: list[tuple[int, float, float, float, float, float]] = []

        for candidate in candidates:
            candidate_box = (candidate[1], candidate[2], candidate[3], candidate[4])
            intersects = any(
                box_iou_xywhn(candidate_box, (saved[1], saved[2], saved[3], saved[4])) > 0.0
                for saved in class_kept
            )
            if not intersects:
                class_kept.append(candidate)

        kept.extend(class_kept)

    return kept


def write_yolo_records(
    records: list[tuple[int, float, float, float, float, float]], label_path: Path
) -> int:
    filtered = nms_same_class(records)
    if not filtered:
        label_path.write_text('', encoding='utf-8')
        return 0

    lines = [
        f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
        for class_id, x_center, y_center, width, height, _score in filtered
    ]
    label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return len(lines)


def write_yolo_boxes_from_model(result, label_path: Path) -> int:
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        label_path.write_text('', encoding='utf-8')
        return 0

    xywhn = boxes.xywhn
    cls_values = boxes.cls if boxes.cls is not None else None
    conf_values = boxes.conf if boxes.conf is not None else None
    records: list[tuple[int, float, float, float, float, float]] = []

    for index, box in enumerate(xywhn):
        class_id = int(cls_values[index].item()) if cls_values is not None else 0
        x_center, y_center, width, height = box.tolist()
        score = float(conf_values[index].item()) if conf_values is not None else 1.0
        records.append((class_id, x_center, y_center, width, height, score))

    return write_yolo_records(records, label_path)


def write_yolo_boxes_from_masks(result, label_path: Path) -> int:
    masks = getattr(result, 'masks', None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        label_path.write_text('', encoding='utf-8')
        return 0

    cls_values = None
    conf_values = None
    boxes = getattr(result, 'boxes', None)
    if boxes is not None and boxes.cls is not None:
        cls_values = boxes.cls
    if boxes is not None and boxes.conf is not None:
        conf_values = boxes.conf

    records: list[tuple[int, float, float, float, float, float]] = []
    for index, mask_tensor in enumerate(masks.data):
        mask_array = mask_tensor.cpu().numpy() > 0.5
        ys, xs = mask_array.nonzero()
        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        mask_h, mask_w = mask_array.shape
        x_center = (x_min + x_max + 1) / (2.0 * mask_w)
        y_center = (y_min + y_max + 1) / (2.0 * mask_h)
        width = (x_max - x_min + 1) / float(mask_w)
        height = (y_max - y_min + 1) / float(mask_h)
        class_id = int(cls_values[index].item()) if cls_values is not None else 0
        score = float(conf_values[index].item()) if conf_values is not None else 1.0

        records.append((class_id, x_center, y_center, width, height, score))

    return write_yolo_records(records, label_path)


def write_yolo_boxes(result, label_path: Path, bbox_source: str) -> int:
    if bbox_source == 'mask':
        return write_yolo_boxes_from_masks(result, label_path)
    return write_yolo_boxes_from_model(result, label_path)


def save_segmentation_preview(result, output_path: Path) -> Path:
    plotted = result.plot(boxes=False, labels=False)
    if getattr(plotted, 'ndim', 0) == 3 and plotted.shape[2] == 3:
        plotted = plotted[:, :, ::-1]
    image = Image.fromarray(plotted)
    image.save(output_path, format='PNG')
    return output_path


def run_prediction(
    image_path: Path,
    model_path: Path,
    targets: list[str],
    conf: float,
    half: bool,
    imgsz: int,
    bbox_source: str,
) -> tuple[Path, Path, int]:
    overrides = {
        'conf': conf,
        'task': 'segment',
        'mode': 'predict',
        'model': str(model_path),
        'half': half,
        'imgsz': imgsz,
        'save': False,
    }
    predictor = SAM3SemanticPredictor(overrides=overrides)

    start_time = time.time()
    total_steps = 5
    current_step = 0

    predictor.set_image(str(image_path))
    current_step += 1
    render_progress(current_step, total_steps, start_time, 'image loaded')

    results = predictor(text=targets)
    current_step += 1
    render_progress(current_step, total_steps, start_time, 'prediction complete')

    preview_path = image_path.with_suffix('.png')
    save_segmentation_preview(results[0], preview_path)
    current_step += 1
    render_progress(current_step, total_steps, start_time, f'saved {preview_path.name}')

    results[0].show(boxes=False, labels=False)
    current_step += 1
    render_progress(current_step, total_steps, start_time, 'shown segmentation')

    label_path = image_path.with_suffix('.txt')
    boxes_written = write_yolo_boxes(results[0], label_path, bbox_source=bbox_source)
    current_step += 1
    render_progress(current_step, total_steps, start_time, f'saved {label_path.name}')
    print()

    return label_path, preview_path, boxes_written


def main() -> None:
    overall_start = time.time()

    parser = argparse.ArgumentParser(
        description='Run SAM3 text-guided prediction and export YOLO bbox labels.'
    )
    parser.add_argument(
        '--input-image',
        type=Path,
        required=True,
        help='Input image filename/path (relative paths are resolved from CWD).',
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        required=True,
        help='Text targets for prediction, e.g. --targets eye nose ear',
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('sam3.pt'),
        help='SAM3 model path (relative paths are resolved from Script Directory).',
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold. Default: 0.25',
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 inference.',
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Inference image size. Auto-aligned to stride. Default: 640',
    )
    parser.add_argument(
        '--stride-align',
        type=int,
        default=14,
        help='Stride multiple used to align --imgsz. Default: 14',
    )
    parser.add_argument(
        '--bbox-source',
        choices=['model', 'mask'],
        default='model',
        help='BBox source for YOLO export: model boxes or mask-derived boxes. Default: model',
    )

    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    image_path = resolve_path(args.input_image, cwd)
    model_path = resolve_path(args.model, script_dir)

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f'Input image not found: {image_path}')
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f'Model not found: {model_path}')
    if args.conf < 0 or args.conf > 1:
        raise ValueError('--conf must be in range [0, 1]')

    aligned_imgsz = align_imgsz(args.imgsz, args.stride_align)

    print(
        'Starting SAM3 prediction to YOLO labels.\n'
        f'- Data root (CWD): {cwd}\n'
        f'- Script dir: {script_dir}\n'
        f'- Input image: {image_path}\n'
        f"- Output label: {image_path.with_suffix('.txt')}\n"
        f"- Output preview image: {image_path.with_suffix('.png')}\n"
        f'- Targets: {args.targets}\n'
        f'- Model: {model_path}\n'
        f'- Confidence: {args.conf}\n'
        f'- Image size: {args.imgsz} (aligned: {aligned_imgsz}, stride: {args.stride_align})\n'
        f'- BBox source: {args.bbox_source}\n'
        f"- NMS: enabled (same-class, any intersection)\n"
        f"- Half precision: {'yes' if args.half else 'no'}"
    )

    label_path, preview_path, boxes_written = run_prediction(
        image_path=image_path,
        model_path=model_path,
        targets=args.targets,
        conf=args.conf,
        half=args.half,
        imgsz=aligned_imgsz,
        bbox_source=args.bbox_source,
    )

    total_runtime = time.time() - overall_start
    print(
        'Done.\n'
        f'- Output preview image: {preview_path}\n'
        f'- Output label file: {label_path}\n'
        f'- Bounding boxes written (after NMS): {boxes_written}\n'
        f'- Total runtime: {total_runtime:.2f}s'
    )


if __name__ == '__main__':
    main()




