# Annotate Images

Runs a YOLO pose model on images and writes YOLO-format pose labels. Images are
loaded from `images` and labels are written to `labels` under the current
working directory (CWD). Model paths are treated as URLs when they contain
`://`; otherwise relative paths are resolved from the script directory. By
default, each image is inferred once (normal). When `--flip` is enabled, each
image is inferred twice (normal and left-right flipped). The flipped results
are mapped back to the original coordinate system, left/right keypoints are
swapped using the flip index map, and keypoint positions are merged with
confidence-weighted averaging before thresholding and formatting.

Objects are matched across normal and flipped predictions using a minimal-cost
assignment. The cost combines confidence-weighted keypoint distance (when
available), box center distance, and box IoU; pairs above the match threshold
are left unmatched and kept as-is.

Progress output includes an ETA clock time when available. If the `images`
folder is missing, the script prints an error and exits. YOLO prediction output
is directed to a temporary project directory so a `runs/` folder is not created
in the dataset root.

## Functions

`annotate_images(root, model_url="yolo11x-pose.pt", img_size=640, conf=0.3, kp_conf=0.4, iou=None, labels_dir_name="labels", flip=False)`

- `root`: Data root (CWD) containing `images` and `labels` folders.
- `model_url`: YOLO pose model path or URL. Relative paths resolve from the
  script directory.
- `img_size`: Inference image size.
- `conf`: Detection confidence threshold passed to `model.predict`.
- `kp_conf`: Keypoint confidence threshold. If a keypoint confidence is below
  this value, its visibility is written as `0` and its `x`/`y` coordinates are
  written as `0.0`. Otherwise, visibility is written as `2`.
- `iou`: IoU threshold passed to `model.predict` for NMS. When `None`, the
  model default is used.
- `labels_dir_name`: Output folder name for labels inside `root`.
- `flip`: When `True`, run flipped inference and merge results.

`write_pose_labels(output_path, pose, kp_conf)`

- `output_path`: Path to the output label file.
- `pose`: Pose data produced from the normal (or merged normal + flipped) inference.
- `kp_conf`: Keypoint confidence threshold used to set visibility and zero
  low-confidence coordinates.

## Flip configuration

- `flip_idx`: `[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`
- If the model does not emit 17 keypoints, the script warns and mirrors
  keypoints without swapping left/right indices.

## Usage

```bash
python annotate_images.py
```

```bash
python annotate_images.py --model yolo11x-pose.pt --img-size 640 --conf 0.3 --kp-conf 0.4 --iou 0.7 --labels labels
```

```bash
python annotate_images.py --flip
```
