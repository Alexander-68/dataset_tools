#!/usr/bin/env python3
"""
Extend images with a flipped duplicate and update YOLO labels/keypoints.

For vertical images (Height > Width), the flipped copy is added to the RIGHT.
For horizontal images (Width >= Height), the flipped copy is added to the BOTTOM.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageOps

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# Keypoint flipping indices for COCO/YOLO standard 17-keypoint skeleton
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def render_progress(current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extend images with a flipped duplicate and update YOLO labels.'
    )
    parser.add_argument('--images-dir', default='images', help='Input images folder.')
    parser.add_argument('--labels-dir', default='labels', help='Input labels folder.')
    parser.add_argument(
        '--images-out', default='images_mosaics', help='Output images folder.'
    )
    parser.add_argument(
        '--labels-out', default='labels_mosaics', help='Output labels folder.'
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ],
        key=lambda p: p.name.lower(),
    )

def read_labels(label_path: Path) -> List[str]:
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]

def format_float(value: float, precision: int = 6) -> str:
    text = f'{value:.{precision}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    if text == '-0':
        return '0'
    return text

def format_vis(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return format_float(value)

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

def process_keypoints(
    parts: List[str],
    is_flipped: bool,
    offset_x_ratio: float,
    offset_y_ratio: float,
    scale_x: float,
    scale_y: float
) -> List[str]:
    """
    Parses, transforms, and reformats keypoints.
    """
    # Keypoints start after class, x, y, w, h (index 5)
    # Each keypoint has 3 values: x, y, visibility
    kp_values = [float(x) for x in parts[5:]]
    num_kpts = len(kp_values) // 3
    
    # Group into triplets
    kpts = []
    for i in range(num_kpts):
        kpts.append((kp_values[i*3], kp_values[i*3+1], kp_values[i*3+2]))

    # If flipping, reorder keypoints according to FLIP_IDX
    if is_flipped:
        # Ensure we don't go out of bounds if kpts count < FLIP_IDX max
        # Standard COCO has 17 kpts. If different, we might skip reordering or warn.
        if len(kpts) == 17:
            kpts = [kpts[i] for i in FLIP_IDX]
        else:
            # Fallback for non-standard counts: just flip coordinates, don't swap IDs
            pass

    out_kpts = []
    for kx, ky, kv in kpts:
        # Invalid keypoints (0,0,0) remain (0,0,0)
        if kx == 0.0 and ky == 0.0 and kv == 0.0:
            out_kpts.extend(['0', '0', '0'])
            continue

        # Flip Logic (Horizontal flip relative to the original image 0..1)
        if is_flipped:
            kx = 1.0 - kx

        # Scale and Offset to placement in new image
        final_x = offset_x_ratio + kx * scale_x
        final_y = offset_y_ratio + ky * scale_y
        
        out_kpts.extend([
            format_float(clamp01(final_x)),
            format_float(clamp01(final_y)),
            format_vis(kv)
        ])
    
    return out_kpts

def process_line(
    line: str,
    is_flipped: bool,
    offset_x_ratio: float,
    offset_y_ratio: float,
    scale_x: float,
    scale_y: float
) -> str:
    parts = line.split()
    class_id = parts[0]
    
    # BBox center x, y, width, height
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    if is_flipped:
        cx = 1.0 - cx  # Horizontal flip of center X

    # Transform BBox
    new_cx = offset_x_ratio + cx * scale_x
    new_cy = offset_y_ratio + cy * scale_y
    new_w = w * scale_x
    new_h = h * scale_y

    out_parts = [
        class_id,
        format_float(clamp01(new_cx)),
        format_float(clamp01(new_cy)),
        format_float(clamp01(new_w)),
        format_float(clamp01(new_h))
    ]

    # Process Keypoints if present
    if len(parts) > 5:
        kpt_strs = process_keypoints(parts, is_flipped, offset_x_ratio, offset_y_ratio, scale_x, scale_y)
        out_parts.extend(kpt_strs)

    return ' '.join(out_parts)

def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    images_dir = resolve_path(Path(args.images_dir), cwd)
    labels_dir = resolve_path(Path(args.labels_dir), cwd)
    images_out = resolve_path(Path(args.images_out), cwd)
    labels_out = resolve_path(Path(args.labels_out), cwd)

    print(
        "Extending images with flipped duplicates.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Images dir: {images_dir}\n"
        f"- Labels dir: {labels_dir}\n"
        f"- Images out: {images_out}\n"
        f"- Labels out: {labels_out}"
    )

    if not images_dir.exists():
        print(f"Error: Images directory '{images_dir}' not found.")
        return 1

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(images_dir)
    if not image_paths:
        print('No images found.')
        return 0

    print(f'Processing {len(image_paths)} images...')

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            with Image.open(img_path) as im:
                img = im.convert('RGB')
            
            w, h = img.size
            is_vertical = h > w
            
            # Create flipped copy
            img_flipped = ImageOps.mirror(img)

            new_lines = []
            label_path = labels_dir / f'{img_path.stem}.txt'
            raw_lines = read_labels(label_path)

            if is_vertical:
                # Vertical: Add flipped to RIGHT
                # New Canvas: 2W x H
                new_w, new_h = w * 2, h
                canvas = Image.new('RGB', (new_w, new_h))
                canvas.paste(img, (0, 0))
                canvas.paste(img_flipped, (w, 0))
                
                # Transformations
                # Original (Left): Scale X by 0.5, Offset X = 0
                # Flipped (Right): Scale X by 0.5, Offset X = 0.5
                
                for line in raw_lines:
                    # Original
                    new_lines.append(process_line(line, False, 0.0, 0.0, 0.5, 1.0))
                    # Flipped
                    new_lines.append(process_line(line, True, 0.5, 0.0, 0.5, 1.0))

            else:
                # Horizontal: Add flipped to BOTTOM
                # New Canvas: W x 2H
                new_w, new_h = w, h * 2
                canvas = Image.new('RGB', (new_w, new_h))
                canvas.paste(img, (0, 0))
                canvas.paste(img_flipped, (0, h))

                # Transformations
                # Original (Top): Scale Y by 0.5, Offset Y = 0
                # Flipped (Bottom): Scale Y by 0.5, Offset Y = 0.5

                for line in raw_lines:
                    # Original
                    new_lines.append(process_line(line, False, 0.0, 0.0, 1.0, 0.5))
                    # Flipped
                    new_lines.append(process_line(line, True, 0.0, 0.5, 1.0, 0.5))

            # Save Image
            out_img_path = images_out / img_path.name
            canvas.save(out_img_path)

            # Save Label
            if new_lines:
                out_lbl_path = labels_out / f'{img_path.stem}.txt'
                out_lbl_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')

        except Exception as e:
            print(f'Error processing {img_path.name}: {e}')
        finally:
            render_progress(idx, len(image_paths))

    if image_paths:
        print()
    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
