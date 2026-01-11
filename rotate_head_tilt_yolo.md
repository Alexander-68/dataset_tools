# Rotate Portraits by Head Tilt (YOLO11 Pose)

This script rotates portrait images around the nose keypoint when the line
between the eyes is nearly horizontal. It updates YOLO11-pose annotations
(bounding boxes and keypoints) and can optionally crop away black borders.
Images that do not meet the tilt threshold are skipped (no output written).

## Logic

1. Read YOLO11-pose labels (`class cx cy w h kpt...`) for each image.
2. Pick the largest bounding box (by area) with class `0` (person) and read its nose/eyes.
3. If the absolute eye-line tilt is under the threshold (default 15 degrees),
   rotate the image clockwise by a random 10–40 degrees around the nose.
4. Apply the same rotation to every object in the label file.
5. Keypoints that fall outside the image are set to `0 0 0`.
6. If cropping is enabled, crop symmetrically based on the rotation expansion:
   remove half of the added height from the top and the same amount from the
   bottom, and remove one-third of the added width from each side.

## Usage

```bash
python rotate_head_tilt_yolo.py [arguments]
```

### Arguments

- `--images-dir`: Input images folder (relative to CWD). Default: `images`
- `--labels-dir`: Input labels folder (relative to CWD). Default: `labels`
- `--images-out`: Output images folder (relative to CWD). Default: `images-r`
- `--labels-out`: Output labels folder (relative to CWD). Default: `labels-r`
- `--no-crop`: Keep the expanded canvas with black borders (disables cropping).
- `--tilt-threshold`: Trigger rotation when tilt is below this value. Default: `15`
- `--rotate-min`: Minimum clockwise rotation angle. Default: `10`
- `--rotate-max`: Maximum clockwise rotation angle. Default: `40`
- `--name-prefix`: Prefix added to rotated outputs. Default: `r_`

### Example

```bash
python rotate_head_tilt_yolo.py --images-dir ./images --labels-dir ./labels \   
  --images-out ./images-r --labels-out ./labels-r
```

## Paths

- Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
- Script directory: contains the script and its `.md` description.

## Requirements

- Python 3
- Pillow (`pip install Pillow`)
