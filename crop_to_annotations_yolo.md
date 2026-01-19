# Crop to Annotations (YOLO)

`crop_to_annotations_yolo.py` crops images and YOLO labels so that all bounding
boxes and visible keypoints fit inside a padded crop with preferred aspect
ratios. It writes cropped JPEG images plus updated labels to new folders.

## Features

- **Annotation-aware crop:** Uses all bounding boxes and visible keypoints to
  build the enclosing rectangle.
- **Ignore tiny boxes:** Objects with a bbox min side under 24 pixels are
  excluded from crop calculations (and their keypoints are ignored).
- **Padding with border clamp:** Adds configurable padding (default 20%), but
  clamps at image borders when needed.
- **Padding fallback:** If a padded crop can't fit a ratio, it retries without
  padding.
- **Aspect ratio search:** Tries 1:1 first; if that fails, tries the largest
  possible square without padding, then 1:2/2:1, 3:4/4:3, and 2:3/3:2.
- **Pair selection:** When both ratios in a pair fit, picks the smaller-area crop.
- **Even sizing:** 1:1, 1:2, and 2:1 crops are normalized to exact, even pixel
  dimensions.
- **Numeric tolerance:** Uses a tiny epsilon when fitting crops to avoid float
  edge-case rejects.
- **Fallback to original:** If no ratio fits, keeps the original image.
- **Prefix by ratio:** Output files get `11_`, `34_`, `43_`, `23_`, `32_`,
  `12_`, `21_`, or `ff_` for non-matching/no-crop cases.
- **Skip empty labels:** If a label file exists but contains no annotations, the
  image and label are not copied.
- **Max dimension:** Resizes output so the longest side is <= `--max-dim`.
- **Skip tiny outputs:** Drops result images and labels if the longest side is
  under 160 pixels.
- **Progress + stats:** Shows a progress bar with ETA and a final summary.

## Usage

```bash
python crop_to_annotations_yolo.py [OPTIONS]
```

### Arguments

- `--images-dir`: Input images folder (relative to CWD). Default: `images`.
- `--labels-dir`: Input YOLO labels folder (relative to CWD). Default: `labels`.
- `--images-out`: Output images folder. Default: `images-crop`.
- `--labels-out`: Output labels folder. Default: `labels-crop`.
- `--padding`: Padding percent around annotations (default: `20`).
- `--max-dim`: Max output image dimension in pixels (default: `1024`, `0` to
  disable).

### Output

Images are saved as JPEGs with the chosen prefix and matching `.txt` labels.
If a label file is missing, the image is still saved but no label is written.

## Paths

- Data root: current working directory (CWD). Relative `--images-dir`,
  `--labels-dir`, `--images-out`, and `--labels-out` paths resolve from CWD.
- Script directory: contains the script and this `.md` description.

## Examples

**1. Default run (CWD images/labels -> images-crop/labels-crop):**

```bash
python crop_to_annotations_yolo.py
```

**2. Use custom folders and 10% padding:**

```bash
python crop_to_annotations_yolo.py --images-dir raw --labels-dir raw_labels --padding 10
```

**3. Disable resizing:**

```bash
python crop_to_annotations_yolo.py --max-dim 0
```

## Dependencies

- `Pillow`

Install dependencies:

```bash
pip install Pillow
```
