# Crop Detected Objects (YOLO)

`crop_detected_objects_yolo.py` crops each image to the union of all labeled
objects (YOLO detection) and visible keypoints (YOLO pose), adds an optional
extra pixel boundary, and rewrites labels to the cropped coordinates.

Default output folders:

- `images-crop`
- `labels-crop`

## Features

- Supports YOLO detection labels (`class cx cy w h`).
- Supports YOLO pose labels (`class cx cy w h x y v ...`).
- Preserves all detected objects; crop is clamped to image borders when needed.
- Optional extra crop boundary in pixels (`--margin-px`, default `20`).
- Rewrites labels for cropped image coordinates.
- Shows progress bar with ETA (`ETA HH:MM`).
- Prints summary stats at the end.

## Usage

```bash
python crop_detected_objects_yolo.py [OPTIONS]
```

## Arguments

- `--images-dir`: Input images folder (relative to CWD). Default: `images`.
- `--labels-dir`: Input labels folder (relative to CWD). Default: `labels`.
- `--images-out`: Output images folder. Default: `images-crop`.
- `--labels-out`: Output labels folder. Default: `labels-crop`.
- `--margin-px`: Extra boundary in pixels around annotation bounds. Default: `20`.

## Path Behavior

- Current Working Directory (CWD) is the data root.
- Relative paths (`--images-dir`, `--labels-dir`, `--images-out`, `--labels-out`)
  resolve from CWD.
- Script Directory is where this script and this `.md` file live.

## Examples

Default run:

```bash
python crop_detected_objects_yolo.py
```

Use a 40-pixel boundary:

```bash
python crop_detected_objects_yolo.py --margin-px 40
```

Use custom folders:

```bash
python crop_detected_objects_yolo.py --images-dir images --labels-dir labels --images-out images-crop --labels-out labels-crop
```

## Dependencies

- `Pillow`

Install:

```bash
pip install Pillow
```
