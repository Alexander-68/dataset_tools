# Delete Similar Frames

`delete_similar_frames.py` removes near-duplicate images in a video frame
sequence by comparing each image to the previous one.

## Behavior

- Scans images in the source folder (default: `images` under CWD).
- Converts frames to grayscale, optionally blurs and downsizes for speed.
- Splits the diff into blocks and deletes the current image only if every block
  is below the threshold (keeps local motion like a moving hand).
- Prints a progress bar with ETA and summary stats.

## Suggested criteria

- Start with `--max-mean-diff 0.01` (1% per-block average pixel difference).
- Tighten to `0.005` for more aggressive pruning of tiny changes.
- Relax to `0.02` if too many frames are removed due to camera noise.
- Use `--block-size 16` (default). Smaller blocks (8) catch small limb motion.
- Use `--downscale 128` or `--downscale 96` for faster runs on large frames.
- Add a small blur (`--blur 0.5` to `1.0`) to ignore compression noise.

## Usage

```bash
python delete_similar_frames.py
```

```bash
python delete_similar_frames.py --source images --max-mean-diff 0.008 --downscale 128
```

```bash
python delete_similar_frames.py --block-size 8 --blur 0.8 --dry-run
```

## Parameters

- `--source`: Source directory containing images. Default: `./images`.
- `--max-mean-diff`: Delete when mean absolute difference is <= this value
  (0..1). Default: `0.01`.
- `--downscale`: Downscale max side before comparison. Use `0` to disable.
  Default: `160`.
- `--block-size`: Block size in pixels for local diff comparison. Default: `16`.
- `--blur`: Gaussian blur radius applied before comparison. Default: `0.0`.
- `--dry-run`: Do not delete files; only report what would be removed.

## Paths

- Data root: current working directory (CWD). Relative input paths resolve from CWD.
- Script directory: contains the script and this `.md` description.

## Requirements

- Python 3
- Pillow
- numpy
