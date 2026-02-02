# Add Face Keypoints (Eyes/Ears)

Adds eye/ear keypoints from a face pose annotation folder into base YOLO pose
labels. Persons are matched by the closest nose keypoint in normalized
coordinates, and only the eye/ear indices are updated.

## Behavior

- Uses `labels` as the base annotations and `labels-face` as the face source.
- Matches each person by the closest valid nose keypoint (index 0).
- Each face annotation is used at most once (greedy nearest-nose matching).
- Only the eye/ear keypoints (indices 1-4) are copied from `labels-face`.
- Nose and other keypoints remain unchanged in the base labels.
- If a match is not found or a face keypoint is missing, the base keypoints
  are left as-is.

## Usage

```bash
python add_face_keypoints.py --labels labels --labels-face labels-face --output labels-merged
```

## Arguments

- `--labels`: Folder with base pose labels (default: `labels`).
- `--labels-face`: Folder with face pose labels (default: `labels-face`).
- `--output`: Output folder for merged labels (default: `labels-merged`).

## Paths

- Data root: current working directory (CWD).
- Script directory: location of this script and its `.md` description.
