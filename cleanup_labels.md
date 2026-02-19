# Cleanup Labels

`cleanup_labels.py` keeps image/label pairs in sync by:
- normalizing filenames in `images` and `labels` (`space` -> `_`)
- moving orphan `.txt` labels (no matching image) to `labels-x`
- creating empty `.txt` labels for images that do not have a label yet

## Behavior

- Scans image files in `images` and label files in `labels` under CWD by default.
- Renames files in `images` and `labels` by replacing spaces with underscores.
- Creates `labels` and `labels-x` if they do not exist.
- Moves orphan label files to `labels-x`.
- Creates empty label files for unlabeled images.
- Prints startup parameters, progress with ETA, and end-of-run stats.

## Parameters

- `--images`: images directory (relative to CWD or absolute). Default: `images`
- `--labels`: labels directory (relative to CWD or absolute). Default: `labels`
- `--labels-x`: target directory for orphan labels. Default: `labels-x`

## Usage

```bash
python cleanup_labels.py
```

```bash
python cleanup_labels.py --images images --labels labels --labels-x labels-x
```

## Paths

- Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
- Script directory: contains the script and this `.md` description.

## Requirements

- Python 3
