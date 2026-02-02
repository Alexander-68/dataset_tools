# Add Face Keypoints

Adds face keypoints from a face pose annotation folder into base YOLO pose
labels. By default, persons are matched by the closest face-area center using
the 5 face keypoints (indices 0-4) with invalid (0,0,0) points ignored. Use
`--nose` to switch to legacy closest-nose matching that only updates eye/ear
keypoints.

## Behavior

- Uses `labels` as the base annotations and `labels-face` as the face source.
- Both folders must exist; the script stops if either is missing.
- Default matching uses the closest face-area center computed from valid face
  keypoints (indices 0-4).
- Each face annotation is used at most once (greedy nearest match).
- All five face keypoints (indices 0-4) are copied as-is from `labels-face`
  when a match is found.
- If a match is not found, exceeds `--max-dist`, or no valid face area exists,
  the base keypoints are left as-is.
- With `--nose`, matches by closest valid nose (index 0) and only updates
  eye/ear keypoints (indices 1-4).

## Usage

```bash
python add_face_keypoints.py --labels labels --labels-face labels-face --output labels-merged --max-dist 0.1
```

Legacy nose matching:

```bash
python add_face_keypoints.py --labels labels --labels-face labels-face --output labels-merged --nose
```

## Arguments

- `--labels`: Folder with base pose labels (default: `labels`).
- `--labels-face`: Folder with face pose labels (default: `labels-face`).
- `--output`: Output folder for merged labels (default: `labels-merged`).
- `--nose`: Use closest-nose matching and update only eye/ear keypoints.
- `--max-dist`: Maximum normalized distance for a match (default: no limit).

## Paths

- Data root: current working directory (CWD).
- Script directory: location of this script and its `.md` description.
