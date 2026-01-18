# Correct Face Keypoints

Merge improved face keypoints into YOLO pose labels. The script uses the
original annotations as the base, then finds matching faces in the
`labels-face` folder and updates the first five keypoints
(nose/eyes/ears) using a weighted blend.

## Behavior

- Uses `labels-orig` as the base annotations and `labels-face` as the face
  source. Output is written to `labels` by default.
- Matches each person by comparing face keypoint areas. It prefers face
  bbox IoU matches, then falls back to center distance.
- If no match is found, the original face keypoints are kept unchanged.
- Face keypoints from `labels-face` have priority: they can create missing
  points or invalidate existing ones.
- When both annotations have a valid keypoint, coordinates are blended using
  `--face-weight` (0.0 = keep original, 1.0 = use face).
- Prints average deviation per face keypoint between original and face inputs.

## Usage

```bash
python correct_face_keypoints.py --labels-orig labels-orig --labels-face labels-face --output labels --face-weight 1.0
```

## Arguments

- `--labels-orig`: Folder with original pose labels (default: `labels-orig`).
- `--labels-face`: Folder with improved face annotations (default: `labels-face`).
- `--output`: Output folder for merged labels (default: `labels`).
- `--face-weight`: Blend weight for face keypoints (0..1, default: 1.0).
- `--match-iou-only`: Match using face bbox IoU only (skip center-distance fallback).
- `--match-iou-threshold`: IoU threshold for matching faces (default: 0.2).
- `--match-center-threshold`: Center distance threshold for matching faces (default: 0.05).

## Paths

- Data root: current working directory (CWD).
- Script directory: location of this script and its `.md` description.
