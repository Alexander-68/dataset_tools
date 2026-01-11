# Extend and Flip Images for YOLO Pose

This script extends images with a horizontally flipped duplicate to augment the dataset. It automatically adjusts YOLO-format bounding boxes and keypoints (pose estimation).

## Logic

The script processes images based on their aspect ratio:

1.  **Vertical Images (Height > Width):**
    *   The image is extended to the **right**.
    *   New size: `2 * Width` x `Height`.
    *   **Left side:** Original image.
    *   **Right side:** Horizontally flipped duplicate.

2.  **Horizontal Images (Width >= Height):**
    *   The image is extended to the **bottom**.
    *   New size: `Width` x `2 * Height`.
    *   **Top side:** Original image.
    *   **Bottom side:** Horizontally flipped duplicate (mirrored left-to-right).

## Label Processing

The script reads YOLO-format label files associated with the images.
Format: `class_id center_x center_y width height [kp1_x kp1_y kp1_vis ...]`

*   **Bounding Boxes:** Coordinates are scaled and shifted to match the new image dimensions.
*   **Keypoints:**
    *   Coordinates are scaled, shifted, and flipped horizontally for the duplicate.
    *   Keypoint indices are swapped to maintain semantic correctness (e.g., left eye becomes right eye on the flipped body).
    *   Swap mapping: `[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]` (Standard COCO 17-keypoint format).
    *   Invalid keypoints `(0,0,0)` are preserved as is.

## Usage

```bash
python extend_flip_yolo.py [arguments]
```

### Arguments

*   `--images-dir`: Path to the input images folder (relative to CWD). Default: `images`
*   `--labels-dir`: Path to the input labels folder (relative to CWD). Default: `labels`
*   `--images-out`: Path to the output folder for processed images (relative to CWD). Default: `images_mosaics`
*   `--labels-out`: Path to the output folder for processed labels (relative to CWD). Default: `labels_mosaics`

### Example

```bash
python extend_flip_yolo.py --images-dir ./dataset/images --labels-dir ./dataset/labels --images-out ./aug_images --labels-out ./aug_labels
```

## Paths

*   Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
*   Script directory: contains the script and its `.md` description.

## Requirements

*   Python 3
*   Pillow (`pip install Pillow`)
