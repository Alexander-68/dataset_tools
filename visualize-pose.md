# Pose Visualization

This script overlays YOLO11/YOLOv8-style pose annotations onto your images so you can quickly verify keypoints.

## Files

- `visualize-pose.py`: Python script that reads images from `images/`, labels from `labels/` (YOLO pose format), and writes visualizations to `visual/`.

## Requirements

- Python 3.8+
- Pillow (`pip install pillow`) if not already available.
- tqdm (`pip install tqdm`) for progress bars.

## Usage

Run from the project root:

```bash
python visualize-pose.py --images images --labels labels --output visual
```

Arguments are optional; defaults point to the folders above.

Optional overrides:

```bash
python visualize-pose.py --dataset-yaml dataset.yaml --keypoint-radius 5
```

## Notes

- Expects YOLO pose labels where each line is `class cx cy w h kp1_x kp1_y kp1_v ...` with normalized coordinates and visibility (v > 0 is drawn).
- Reads `kpt_shape` from `dataset.yaml` to determine the expected number of keypoints (defaults to 17 if missing).
- Uses a COCO-style 17-keypoint skeleton; if fewer points are present, lines that reference missing points are skipped.
- Bounding boxes are drawn in the same color as their keypoints; missing label files are skipped with a message.
