# Rotate Images + Labels Script

`rotate_images_labels.py` rotates images and their YOLO detection/pose labels by
90, 180, and 270 degrees (clockwise), writing the rotated outputs to new
folders. It can optionally rotate only selected angles and adds a prefix to
output filenames.

## Features

-   **Rotation Augmentation:** Rotates images by fixed 90/180/270 degrees.
-   **Label Support:** Updates YOLO detection or YOLO pose labels.
-   **Separate Outputs:** Writes rotated images/labels to separate folders.
-   **Progress + Stats:** Shows a progress bar and prints summary stats.

## Usage

```bash
python rotate_images_labels.py [OPTIONS]
```

### Arguments

-   `--images-dir`: Input images folder (relative to CWD). Default: `images`.
-   `--labels-dir`: Input labels folder (relative to CWD). Default: `labels`.
-   `--images-out`: Output images folder. Default: `images-rot`.
-   `--labels-out`: Output labels folder. Default: `labels-rot`.
-   `--angles`: Comma-separated clockwise angles. Default: `90,180,270`.
-   `--prefix`: Prefix for output filenames. Default: `r_`.

### Output

-   Images are saved as: `{prefix}{angle}_{original_name}`
-   Labels are saved as: `{prefix}{angle}_{original_stem}.txt`

At startup, the script prints what it will do and the parameters. While running,
it shows a progress bar. At the end, it prints stats for images and labels.

### Examples

**1. Rotate all images by 90/180/270 (default):**

```bash
python rotate_images_labels.py
```

**2. Rotate only 90 and 270 degrees:**

```bash
python rotate_images_labels.py --angles 90,270
```

**3. Custom output folders and filename prefix:**

```bash
python rotate_images_labels.py --images-out images_r --labels-out labels_r --prefix rot_
```

## Paths

-   Data root: current working directory (CWD). Relative input/output paths
    resolve from CWD.
-   Script directory: contains the script and its `.md` description.

## Dependencies

-   `Pillow` (PIL)

To install dependencies:

```bash
pip install Pillow
```
