# Dataset Tile Optimizer (YOLO)

Builds an optimized dataset by tiling images into mosaics and rewriting YOLO
labels (detection or pose). Small images are combined into larger grids to
increase instances per image and reduce low-resolution upscaling during
training. Input/output paths are resolved from the current working directory
(CWD).

## Buckets and tiling rules

- Square images <= 224 px: shuffled and tiled into 3x3 mosaics (`t9_`), then
  shuffled again and tiled into 3x3 mosaics with random 0/90/270 rotations per
  tile (`t9r_`).
- Square images <= 384 px: shuffled into 2x2 mosaics (`t4_`), then shuffled
  again with per-tile rotation (`t4r_`).
- Square images > 384 px: copied as-is, then shuffled into 2x2 mosaics (`t4_`)
  and rotated-tile mosaics (`t4r_`).
- Aspect 1:2 or 9:16: shuffled into 2x1 mosaics (`t2_`), then the mosaic is
  rotated by 90 or 270 degrees (`t2r_`).
- Aspect 2:1 or 16:9: shuffled into 1x2 mosaics (`t2_`), then rotated into
  `t2r_`.
- Aspect 2:3 or 3:2: copied as-is, then tiled into 3x2 or 2x3 mosaics (`t6_`)
  and rotated into `t6r_`.
- Remaining images: copied as-is, then tiled into `t4_` and `t4r_` mosaics.

If the final group is incomplete, images from the start of the same shuffled
sequence are reused to fill the grid.

## Resizing rules

- Tile size is chosen per mosaic as the maximum tile size in that group; smaller
  tiles are scaled up to fit.
- Images are resized to fit within each tile while preserving aspect ratio, with
  black padding.
- If `--max-dim` is set and an output exceeds it, the output is scaled down while
  preserving aspect ratio. YOLO labels remain valid because they are normalized.

## Labels

- Detection format: `class cx cy w h`
- Pose format: `class cx cy w h x1 y1 v1 ...`
- Missing label files are treated as empty.
- Keypoints with `v <= 0` are written as `0 0 0` after transforms.

## Output naming

Mosaics use the specified prefix and a 6-digit counter, e.g. `t4_000012.jpg`.
Rotated variants reuse the same counter (e.g. `t2_000010.jpg` and
`t2r_000010.jpg`).

## Arguments (defaults)

- `--images-dir images`: input images folder (relative to CWD).
- `--labels-dir labels`: input labels folder (relative to CWD).
- `--images-out images-opt`: output images folder (relative to CWD).
- `--labels-out labels-opt`: output labels folder (relative to CWD).
- `--max-dim 1024`: scale outputs down if max dimension exceeds this value; use
  `0` to disable resizing.
- `--out-ext .jpg`: output extension for mosaics.

## Usage

```bash
python optimize_dataset_tiles_yolo.py
```

```bash
python optimize_dataset_tiles_yolo.py --max-dim 1024
```

```bash
python optimize_dataset_tiles_yolo.py --images-dir images --labels-dir labels --images-out images-opt --labels-out labels-opt
```

```bash
python optimize_dataset_tiles_yolo.py --out-ext .png
```

## Paths

- Data root: current working directory (CWD). Relative input/output paths resolve
  from CWD.
- Script directory: contains the script and its `.md` description.

## Requirements

Python 3 and Pillow:

```bash
pip install pillow
```
