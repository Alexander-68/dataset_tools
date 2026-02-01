# Square Portrait Crop (YOLO Pose)

`crop_portrait_square_yolo.py` crops portrait images to a square (or optional
2:3) by centering on the face keypoints (nose, eyes, ears) detected by a YOLO
pose model.

## Features

- **Face-centered crop:** Computes the face center as the centroid of keypoints
  0–4 (nose, eyes, ears) and crops a square around that point.
- **Largest valid person:** When multiple people are detected, selects the
  largest `person` object with valid face keypoints.
- **Two-person crop:** If two large people with valid face keypoints are
  detected, the crop expands to include both faces using the face centers and
  diameters (face circles). "Large" means each person is at least 2% of the
  image area, and the second person is at least 45% of the largest person.
- **In-place or destination:** Overwrite the original images or write to a new
  output folder.
- **Optional resize:** Resize the final crop to a target width (height follows the aspect).
- **Face diameter crop:** Optionally size the crop using face diameter.
- **Debug overlay:** Draw a white face circle before cropping.
- **Rotation:** Optional random clockwise rotation before cropping.
- **2:3 ratio:** Optional portrait ratio crop (width:height = 2:3).
- **Flip:** Optional left-to-right flip after cropping.
- **EXIF orientation:** Respects EXIF rotation via auto-transpose.
- **Progress + stats:** Shows a progress bar (with ETA) and final summary.

## Usage

```bash
python crop_portrait_square_yolo.py [OPTIONS]
```

### Arguments

- `--input`: Input images folder (relative to CWD). Defaults to `images`.
- `--output`: Optional output folder. If omitted, crops in-place.
- `--model`: YOLO pose model path/name. Relative paths resolve from the script
  directory. Defaults to `yolo11x-pose.pt`.
- `--size`: Optional output width (pixels) for the final crop (height follows aspect).
- `--crop-percent`: Expand crop size from face diameter by a percent (default: max square).
- `--rotate`: Random clockwise rotation between 10 and 45 degrees before cropping.
- `--ratio-23`: Use a 2:3 (width:height) crop instead of square.
- `--flip`: Flip the final cropped image left-to-right.
- `--debug-draw`: Draw a white circle (face center + diameter) before cropping.

### Output

The script prints a progress bar with an ETA clock time while processing and a
summary of how many images were cropped, skipped, or had no pose detected.

## Paths

- Data root: current working directory (CWD). Relative `--input` and `--output`
  paths resolve from CWD.
- Script directory: contains the script, description, and default model files.

### Examples

**1. Crop images in `images/` in-place (default behavior):**

```bash
python crop_portrait_square_yolo.py
```

**2. Crop images in `raw_photos/` in-place:**

```bash
python crop_portrait_square_yolo.py --input raw_photos
```

**3. Crop images and save to `cropped/`:**

```bash
python crop_portrait_square_yolo.py --input raw_photos --output cropped
```

**4. Use a different YOLO pose model:**

```bash
python crop_portrait_square_yolo.py --model yolo11s-pose.pt
```

**5. Resize the final crop to 512x512:**

```bash
python crop_portrait_square_yolo.py --size 512
```

**6. Crop using face diameter + 20% and draw debug circle:**

```bash
python crop_portrait_square_yolo.py --crop-percent 20 --debug-draw
```

**7. Rotate clockwise randomly (10-45 degrees) before cropping:**

```bash
python crop_portrait_square_yolo.py --rotate
```

**8. Flip the final crop left-to-right:**

```bash
python crop_portrait_square_yolo.py --flip
```

**9. Crop with 2:3 ratio and resize to 512x768:**

```bash
python crop_portrait_square_yolo.py --ratio-23 --size 512
```

## Dependencies

- `ultralytics`
- `Pillow`

Install dependencies:

```bash
pip install ultralytics Pillow
```
