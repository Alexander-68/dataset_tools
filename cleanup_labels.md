# Cleanup Labels

`cleanup_labels.py` moves YOLO label files that do not have a matching image
into a separate folder.

## Behavior

- Scans `labels` and `images` under the current working directory (CWD).
- Moves orphaned label files into `labels-x`.
- Prints a progress bar and a final count of moved files.

## Usage

```bash
python cleanup_labels.py
```

## Paths

- Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
- Script directory: contains the script and this `.md` description.

## Requirements

- Python 3
