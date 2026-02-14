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


def write_yolo_boxes(result, label_path: Path) -> int:
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        label_path.write_text('', encoding='utf-8')
        return 0

    xywhn = boxes.xywhn
    cls_values = boxes.cls if boxes.cls is not None else None
    lines: list[str] = []

    for index, box in enumerate(xywhn):
        class_id = int(cls_values[index].item()) if cls_values is not None else 0
        x_center, y_center, width, height = box.tolist()
        lines.append(
            f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
        )

    label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return len(lines)


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
) -> tuple[Path, Path, int]:
    overrides = {
        'conf': conf,
        'task': 'segment',
        'mode': 'predict',
        'model': str(model_path),
        'half': half,
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
    boxes_written = write_yolo_boxes(results[0], label_path)
    current_step += 1
    render_progress(current_step, total_steps, start_time, f'saved {label_path.name}')
    print()

    return label_path, preview_path, boxes_written


def main() -> None:
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
        f"- Half precision: {'yes' if args.half else 'no'}"
    )

    label_path, preview_path, boxes_written = run_prediction(
        image_path=image_path,
        model_path=model_path,
        targets=args.targets,
        conf=args.conf,
        half=args.half,
    )

    print(
        'Done.\n'
        f'- Output preview image: {preview_path}\n'
        f'- Output label file: {label_path}\n'
        f'- Bounding boxes written: {boxes_written}'
    )


if __name__ == '__main__':
    main()
