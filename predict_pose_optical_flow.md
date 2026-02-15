# Predict Pose Keypoints with Optical Flow

Predict YOLO-pose keypoints for target indexed images by tracking keypoints
from a previous indexed image that already has matching YOLO annotations in
`labels/`. The script copies the source bbox and writes predicted keypoints in
YOLO pose format. If a target label file already exists, predicted object
annotation(s) are appended.

## Target Selection

- Single mode: `--target-index N`
- If index `N` is missing in single mode, the script uses the next available indexed image.
- Range mode: `--target-index START END`
- Range mode processes all available indexed images in `START..END`.
- If `END` is larger than available indices, processing continues up to the last available index.
- If no usable target image exists, the script exits cleanly with an error message (no traceback crash).

## Source Selection and Output Rules

- For each target image, the script searches backward for the nearest previous indexed image with a matching label file and class/object requirement.
- Default source object is `0`; use `--source-object -1` to propagate all matching class objects.
- Source bbox is duplicated; keypoints are replaced with predicted values.
- If target label file exists, predicted object line(s) are appended.
- Source-invalid keypoints (`0 0 0` or visibility `0`) stay `0 0 0`.
- If tracking fails after retry, the script keeps the previous source keypoint position.
- Range mode uses compact output: one batch progress bar and final aggregated stats.

## Usage

```bash
python predict_pose_optical_flow.py --target-index 125
```

Process a range of images:

```bash
python predict_pose_optical_flow.py --target-index 15 100
```

Propagate all person objects from the previous labeled frame:

```bash
python predict_pose_optical_flow.py --target-index 125 --source-object -1
```

## Arguments

- `--target-index`: Required. Pass one value `N` for single mode, or two values `START END` for range mode.
- `--images`: Images folder (default: `images`).
- `--labels`: Labels folder (default: `labels`).
- `--source-object`: Source object index in previous label file (0-based). Use `-1` for all matching objects. Default: `0`.
- `--class-id`: Class id to propagate. Default: `0`.
- `--num-keypoints`: Number of keypoints to write in YOLO pose output. Default: `17`.
- `--backcheck-threshold`: Max backward-forward mismatch in pixels for accepting track. Default: `1.5`.
- `--window-size`: LK optical-flow window size in pixels (square). Default: `32`.

## Optical Flow Method

The script tracks each keypoint independently using sparse Pyramidal
Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) from frame `N-1` to
frame `N`. For robustness, it performs a forward-backward consistency check:
after forward tracking, the predicted point is tracked back, and the
round-trip error must stay under `--backcheck-threshold`. If the first pass
fails, the script retries once with a larger window and a relaxed back-check
limit, then falls back to keeping the previous keypoint position.

## Optical Flow Parameters

- Method: `cv2.calcOpticalFlowPyrLK`
- Pyramid levels: `3`
- Window size: `W x W`, where `W` is `--window-size` (default `32`)
- Termination criteria: `max_count=30`, `epsilon=0.01`
- Back-check: forward `(N-1 -> N)` then reverse `(N -> N-1)`
- Retry on failure: window `+50%` (for default `32`, retry uses `48x48`), back-check threshold `x2`

## Paths

- Data root: current working directory (CWD).
- Script directory: location of this script and its `.md` description.
