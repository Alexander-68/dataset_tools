# YOLO Pose to COCO JSON

`yolo_pose_to_coco_json.py` converts YOLO11-pose labels into COCO JSON files.
It expects `images/` and `labels/` with `train/` and `val/` subfolders and
writes one COCO JSON per split.

## Features

-   **Train/Val Support:** Reads `images/<split>` and `labels/<split>`.
-   **COCO Output:** Builds `images`, `annotations`, and `categories` sections.
-   **Keypoints:** Converts normalized YOLO pose keypoints to COCO pixel values.
-   **Progress + Stats:** Shows a progress bar with ETA and prints summary stats.

## Usage

```bash
python yolo_pose_to_coco_json.py [OPTIONS]
```

### Arguments

-   `--images-dir`: Images root folder (default: `images`).
-   `--labels-dir`: Labels root folder (default: `labels`).
-   `--splits`: Comma-separated splits (default: `train,val`).
-   `--output-dir`: Output folder for JSON files (default: `coco`).
-   `--dataset-yaml`: Dataset YAML with keypoint/class names (default: `dataset.yaml`).
-   `--vis-threshold`: Visibility threshold for 0-1 keypoint values (default: `0.5`).

### Output

-   JSON files are written to: `coco/coco_train.json`, `coco/coco_val.json`
-   `file_name` values are relative to the images root, e.g. `train/img_001.jpg`.

At startup, the script prints what it will do and the parameters. While running,
it shows a progress bar. At the end, it prints per-split and total stats.

## Examples

**1. Convert default train/val splits:**

```bash
python yolo_pose_to_coco_json.py
```

**2. Custom folders and output location:**

```bash
python yolo_pose_to_coco_json.py --images-dir data/images --labels-dir data/labels --output-dir annotations
```

**3. Convert only a single split:**

```bash
python yolo_pose_to_coco_json.py --splits train
```

## Paths

-   Data root: current working directory (CWD). Relative input/output paths
    resolve from CWD.
-   Script directory: contains the script and its `.md` description (and
    `dataset.yaml` by default).

## Dependencies

-   `Pillow` (PIL)

To install dependencies:

```bash
pip install Pillow
```
