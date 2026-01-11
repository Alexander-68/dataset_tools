# Self Mosaic YOLO Builder

Builds a self-mosaic by extending the short axis by `mosaic_scale * long_axis` and
rewrites YOLO labels (detection-only or pose with 17 keypoints). The script
reads images from `images` and labels from `labels` under the current working
directory (CWD), and writes mosaics to `images_mosaic` and `labels_mosaic` by
default. Output file names use the `--prefix` value (default `s_`).

## Layout rules

- Landscape (width > height): canvas is `width x (height + width * mosaic_scale)`.
- Portrait (height >= width): canvas is `(width + height * mosaic_scale) x height`.
- The extra dimension uses `int(long_axis * mosaic_scale)` pixels.
- The original image is placed at `(0, 0)` without flipping.
- Remaining space is filled with 5 more tiles aligned to the top-left of the
  current remaining rectangle (no cropping).
- The canvas background is black.

## Mosaic placement rules

Landscape (width > height):

- Step 2: rotate 90 degrees CCW, scale to fit the remaining height, place at
  the remaining top-left.
- Step 3: scale to fit the remaining width, flip left-right, place at the
  remaining top-left.
- Step 4: rotate 270 degrees CCW, scale to fit the remaining height, place at
  the remaining top-left.
- Step 5: scale to fit the remaining width, place at the remaining top-left.
- Step 6: scale to fit the remaining width, flip left-right, place at the
  remaining top-left.

Portrait (height >= width): follow the same steps, but swap remaining width
and remaining height in the scale-to-fit logic. The remaining space starts at
`(width, 0)`.

## Labels

Expected YOLO formats:

- Detection: `class cx cy w h`
- Pose: `class cx cy w h x1 y1 v1 ... x17 y17 v17`

Notes:

- Output labels are normalized to the final mosaic size.
- Labels are duplicated for each placement until a placement yields no labels
  after filtering.
- Any box that ends up smaller than `--min-bbox-pixels` (default 20) in width
  or height after a placement transform is dropped.
- Keypoints with `(0, 0, 0)` are preserved as-is.
- Keypoint visibility `v` is unchanged.
- Missing label files are treated as empty.
- Tiles are added in order; if a placement yields no labels after filtering,
  remaining tiles are skipped and left black.
- If no labels remain for the original placement, no image or label file is
  written for that input.

## Flip keypoints

The Step 3 and Step 6 tiles are flipped left-right. Keypoints are re-ordered
using this index map (zero-based):

`flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`

## Output naming

Each input image with at least one remaining label writes:

- Image: `<prefix><stem>.jpg` (or `--out-ext`)
- Label: `<prefix><stem>.txt`

Default prefix is `s_` (use `--prefix ""` to disable).

## Usage

```bash
python mosaic_self_yolo.py
```

```bash
python mosaic_self_yolo.py --max-dim 1024
```

```bash
python mosaic_self_yolo.py --mosaic-scale 0.7
```

```bash
python mosaic_self_yolo.py --min-bbox-pixels 32
```

```bash
python mosaic_self_yolo.py --images-dir images --labels-dir labels --images-out images_mosaic --labels-out labels_mosaic
```

```bash
python mosaic_self_yolo.py --prefix ""
```

## Paths

- Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
- Script directory: contains the script and its `.md` description.

## Requirements

Python 3 and Pillow:

```bash
pip install pillow
```
