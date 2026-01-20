# Resize Images Script

`resize_images.py` is a utility script to resize images in a directory. It supports resizing based on a maximum dimension and can convert images to JPEG.

## Features

-   **Resizing:** Resizes images so that their longest side does not exceed a specified maximum dimension.
-   **HEIC Support:** Converts HEIC/HEIF images to JPEG during processing.
-   **Force JPEG:** Optionally save all outputs as `.jpg` regardless of input format.
-   **In-Place or Destination:** Can modify images in-place (overwriting originals) or save processed images to a new destination folder.
-   **EXIF Orientation:** Respects EXIF orientation data (autorotate).
-   **Progress + Stats:** Prints a progress bar while processing and reports total size before/after.

## Usage

```bash
python resize_images.py [OPTIONS]
```

### Arguments

-   `--source`: Path to the directory containing images to process (relative to CWD). Defaults to `images`.
-   `--destination`: Optional path to a destination directory.
    -   If provided, processed images are saved to this directory. The source directory is left unchanged.
    -   If **not** provided, the script operates **in-place**. Images are resized/converted and overwritten in the source directory. **Original HEIC files are deleted** after successful conversion to JPEG in in-place mode.
-   `--size`: The maximum dimension (width or height) for the resized images. Defaults to `1024` pixels.
-   `--force-jpg`: Save all output images as JPEG (`.jpg`), converting formats as needed. In in-place mode, non-JPEG originals are replaced.

### Output
At startup, the script prints what it will do and the parameter values. During processing, it shows a progress bar. When finished, it reports counts plus total size before and after (for processed images).

### Examples

**1. Resize images in `images/` to 1024px in-place (default behavior):**

```bash
python resize_images.py
```

**2. Resize images in `raw_photos/` to 2048px in-place:**

```bash
python resize_images.py --source raw_photos --size 2048
```

**3. Resize images in `raw_photos/` and save them to `processed_photos/`:**

```bash
python resize_images.py --source raw_photos --destination processed_photos
```

**4. Resize to a smaller thumbnail size (e.g., 512px):**

```bash
python resize_images.py --size 512
```

**5. Convert all images to JPEG while resizing:**

```bash
python resize_images.py --force-jpg
```

## Paths

-   Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
-   Script directory: contains the script and its `.md` description.

## Dependencies

-   `Pillow` (PIL)
-   `pillow-heif` (optional, for HEIC support)

To install dependencies:

```bash
pip install Pillow pillow-heif
```
