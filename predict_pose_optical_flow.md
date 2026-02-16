# Predict Pose Keypoints with Optical Flow

Predict YOLO-pose keypoints for target indexed images by tracking keypoints
from previous indexed images that already have matching YOLO annotations in
`labels/`. The script copies bbox values from the nearest source frame and
writes predicted keypoints in YOLO pose format. If a target label file already
exists, predicted object annotation(s) are appended.

By default, the script uses **multi-frame source fusion** to reduce temporal
drift ("sliding" keypoints on smooth textures).

## Target Selection

- Single mode: `--target-index N`
- If index `N` is missing in single mode, the script uses the next available indexed image.
- Range mode: `--target-index START END`
- Range mode processes all available indexed images in `START..END`.
- If `END` is larger than available indices, processing continues up to the last available index.
- If no usable target image exists, the script exits cleanly with an error message (no traceback crash).

## Source Selection and Output Rules

- For each target image, the script searches backward for previous indexed images with a matching label file and class/object requirement.
- Source strategy is controlled by `--history-mode`:
  - `all` (default): always look backward (full history scope), capped by `--history-size`.
  - `batch`: do not look backward outside the current processed batch range.
- `--history-size` limits how many previous labeled frames are used per target.
- In `batch` mode, history grows forward inside the current run/batch.
- The first target in `batch` mode may be skipped if there is no valid source within batch.
- `--history-size` is capped at `16` (max). Default is `16`.
- `--history-size 0` is invalid.
- Source object selection is newest-first: `--source-object 0` means the last matching annotation line in the source label file (most recent), `1` means second last.
- Use `--source-object -1` to propagate all matching class objects.
- Source bbox is copied from the nearest source frame; keypoints are replaced with predicted/fused values.
- If target label file exists, predicted object line(s) are appended.
- Source-invalid keypoints (`0 0 0` or visibility `0`) stay `0 0 0`.
- If tracking fails after retry, the script keeps the previous source keypoint position.
- Range mode uses compact output: one batch progress bar and final aggregated stats.
- `ETA` in progress bars is shown as estimated finish clock time (local), not remaining duration.

## Usage

```bash
python predict_pose_optical_flow.py --target-index 125
```

Process a range of images:

```bash
python predict_pose_optical_flow.py --target-index 15 100
```

Use in-batch-only history:

```bash
python predict_pose_optical_flow.py --target-index 15 100 --history-mode batch
```

Propagate all person objects from the previous labeled frame:

```bash
python predict_pose_optical_flow.py --target-index 125 --source-object -1
```

Use bounded history fusion:

```bash
python predict_pose_optical_flow.py --target-index 15 100 --history-mode all --history-size 12 --history-decay 10
```

## Arguments

- `--target-index`: Required. Pass one value `N` for single mode, or two values `START END` for range mode.
- `--images`: Images folder (default: `images`).
- `--labels`: Labels folder (default: `labels`).
- `--source-object`: Source object index in previous label file, newest-first (0-based). `0` means last matching line, `1` means second last. Use `-1` for all matching objects. Default: `0`.
- `--class-id`: Class id to propagate. Default: `0`.
- `--num-keypoints`: Number of keypoints to write in YOLO pose output. Default: `17`.
- `--backcheck-threshold`: Max backward-forward mismatch in pixels for accepting track. Default: `1.5`.
- `--window-size`: LK optical-flow window size in pixels (square). Default: `32`.
- `--history-mode`: Source strategy: `all` (always look backward) or `batch` (restrict to current batch history). Default: `all`.
- `--history-size`: Max previous labeled source frames per target (max `16`, min `1`). `1` means previous image only. Default: `16`.
- `--history-size 0` is invalid.
- `--history-decay`: Recency decay in frames for fusion weighting. Higher values keep older frames influential longer. Default: `8.0`.

## Optical Flow Method

For each source frame, the script tracks each keypoint independently using
sparse Pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) from
source frame to target frame.

Robustness steps:

- Forward-backward consistency check (round-trip error must pass threshold).
- One retry with larger window and relaxed back-check threshold.
- In `all` mode, successful candidates from multiple source frames are fused
  using recency + quality weighting with outlier rejection.
- If no candidate succeeds, the script falls back to the nearest valid source keypoint.

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
