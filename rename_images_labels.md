# Rename Images + Labels Script

`rename_images_labels.py` renames images in a folder and their corresponding
label files to a sequential series. It is designed for datasets that store
images in `images/` and YOLO-style labels in `labels/` with matching stems.

## Features
-   **Sequential Rename:** Renames images to `PREFIX0001` style names while
    preserving each image's extension.
-   **Label Sync:** Renames label files in `labels/` that match each image stem
    (expects `.txt` labels). If the labels folder is missing, labels are skipped.
-   **Prefix Support:** Optional `--prefix` lets you add a prefix to every new
    filename (include your own separator if desired).
-   **Extension Normalize:** Lowercases image extensions (e.g., `.JPG` to `.jpg`)
    during renaming.
-   **Progress + Stats:** Prints a progress bar and a summary at the end.

## Usage
```bash
python rename_images_labels.py [OPTIONS]
```

### Arguments
-   `--images`: Images directory (relative to CWD). Defaults to `images` if it
    exists.
-   `--labels`: Labels directory (relative to CWD). Defaults to `labels` if it
    exists. If missing, label renames are skipped.
-   `--prefix`: Optional prefix for new filenames (e.g., `train_`). Defaults to
    empty.
-   `--start`: Starting index for the sequence. Defaults to `1`.

### Output
At startup, the script prints what it will do and all parameter values. During
processing, it shows a progress bar. When finished, it reports counts for images
renamed, skipped, labels renamed, and labels missing.

### Examples

**1. Rename images and labels in the default folders:**
```bash
python rename_images_labels.py
```

**2. Use a prefix and start at 1000:**
```bash
python rename_images_labels.py --prefix train_ --start 1000
```

**3. Use custom folders:**
```bash
python rename_images_labels.py --images data/images --labels data/labels
```

## Paths
-   Data root: current working directory (CWD). Relative input paths resolve
    from CWD.
-   Script directory: contains the script and its `.md` description.
