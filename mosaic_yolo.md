# Mosaic YOLO Builder

Builds mosaics from image groups and merges YOLO labels
(detection-only or pose with 17 keypoints). The script reads images from
`images` and labels from `labels` under the current working directory (CWD),
randomizes the input list once at startup, and writes mosaics to
`images_mosaic` and `labels_mosaic` by default. Progress is displayed with a
terminal progress bar.

<p align="center">
  <img width="45%" hspace="10" alt="mosaic_ex" src="https://github.com/user-attachments/assets/9db1bd5a-0827-4fb2-b4c9-931d8ef8e7d0" />
  <img width="45%" hspace="10" alt="mosaic_ex2" src="https://github.com/user-attachments/assets/f0bc0501-c471-4654-ba3e-91cf14a1ed59" />
</p>

## Layout rules

- Images are treated as square if `min_dim / max_dim > 0.8`.
- When `--rotate` is off:
  - Square images are combined in 2x2 grids.
  - Portrait (vertical) images are combined in 3x2 grids.
  - Landscape (horizontal) images are combined in 2x3 grids.
  - Groups are formed only within the same shape category (no mixing square,
    vertical, and horizontal).
- When `--rotate` is on:
  - All images are combined in 2x2 grids regardless of orientation.

## Resizing rules

- Original sizes are preserved when possible.
- If dimensions do not match, images within each row are resized to match the
  first image height. Each row is uniformly scaled to match the first row width.

## Labels

Expected YOLO formats:

- Detection: `class cx cy w h`
- Pose: `class cx cy w h x1 y1 v1 ... x17 y17 v17`

Notes:

- Output labels are normalized to the final mosaic size.
- Any box that ends up smaller than 20 pixels in width or height is dropped.
- Keypoints with `(0, 0, 0)` are preserved as-is.
- Keypoint visibility `v` is unchanged.
- Missing label files are treated as empty.

## Flip

Use `--flip` to mirror each source image left-right before composing mosaics.
Bounding box centers and keypoint `x` values are mirrored. Keypoints are
re-ordered using this index map (zero-based):

`flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`

## Rotate

Use `--rotate` to randomly rotate each source image before composing mosaics.   
Each image has a 0.25 probability of a 90-degree rotation, 0.25 probability of  
a 270-degree rotation, and 0.5 probability of no rotation. Labels are rotated  
to match.

## Output naming

Mosaics are grouped sequentially in the randomized order within each           
orientation. The output filename uses the 1-based sequence numbers from the     
overall randomized input list:

-- Image: `<n>_<m>_<k>_<q>_<r>_<s>.jpg` (or `--out-ext`) for vertical and
  horizontal mosaics when `--rotate` is off, `<n>_<m>_<k>_<q>.jpg` for square
  mosaics or any mosaic when `--rotate` is on.
- Label: `<n>_<m>_<k>_<q>_<r>_<s>.txt` or `<n>_<m>_<k>_<q>.txt`

Use `--name-prefix` to prepend a string to the generated output filenames.

If there are unpaired images in either orientation, they are completed with     
random previously used images of the same orientation to complete the mosaic.   

## Usage

```bash
python mosaic_yolo.py
```

```bash
python mosaic_yolo.py --max-dim 1024
```

```bash
python mosaic_yolo.py --flip
```

```bash
python mosaic_yolo.py --rotate
```

```bash
python mosaic_yolo.py --name-prefix aug_
```

```bash
python mosaic_yolo.py --images-dir images --labels-dir labels --images-out images_mosaic --labels-out labels_mosaic
```

## Paths

- Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
- Script directory: contains the script and its `.md` description.

## Requirements

Python 3 and Pillow:

```bash
pip install pillow
```
