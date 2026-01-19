# Correct MPII Keypoints

Correct selected COCO keypoints in YOLO pose labels using MPII human pose
annotations. The script matches each person by checking how many MPII
keypoints fall inside the YOLO person bounding box, then replaces shoulders,
elbows, wrists, hips, knees, and ankles using the MPII values.

## Behavior

- Uses `labels-x` as the base annotations and `labels-mpii` as the MPII source.
  Output is written to `labels` by default.
- A match requires most MPII keypoints to be inside the YOLO bbox; the threshold
  is configurable with `--match-threshold`.
- If no match is found or MPII labels are missing, the original keypoints are
  kept.
- Replaces the following COCO keypoints: left/right shoulders, elbows, wrists,
  hips, knees, and ankles.
- Any keypoints that fall outside the corresponding YOLO person bbox are cleared
  (set to 0,0,0).
- Prints replacement statistics per keypoint.

## Usage

```bash
python correct_mpii_keypoints.py --labels-x labels-x --labels-mpii labels-mpii --output labels
```

## Arguments

- `--labels-x`: Folder with YOLO pose labels to correct (default: `labels-x`).
- `--labels-mpii`: Folder with MPII pose labels (default: `labels-mpii`).
- `--output`: Output folder for merged labels (default: `labels`).
- `--match-threshold`: Minimum ratio of MPII keypoints inside the YOLO bbox to
  accept a match (default: 0.5).

## Keypoint Mapping

- MPII joints replaced: right/left ankle, knee, hip, shoulder, elbow, wrist.
- COCO indices follow the standard YOLO pose order: `nose` through `right_ankle`.

## Paths

- Data root: current working directory (CWD).
- Script directory: location of this script and its `.md` description.
