# merge_datasets.py

`merge_datasets.py` consolidates multiple pill-detection datasets into a single training/validation layout that matches the structure expected in `Verified/Final_dataset`. It also documents the image counts per source inside `dataset.yaml`.

## Dataset Layout

```
Verified/
|-- Final_dataset/
|   |-- images/
|   |   |-- train/
|   |   `-- val/
|   `-- labels/
|       |-- train/
|       `-- val/
`-- <other-dataset>/
    |-- images/
    `-- labels/
```

- Each source dataset inside `Verified/` (other than `Final_dataset`) must contain matching `images/` and `labels/` directories.
- Label files share their image name but use a `.txt` extension.

## How It Works

1. Ensures the `train` and `val` folders exist under the target `images/` and `labels/` directories.
2. Iterates through every source dataset in `Verified/`, skipping the target folder itself.
3. Validates that each image has a matching `.txt` label file.
4. Copies image/label pairs into the Pill dataset:
   - If a source dataset has fewer than 64 images, all pairs are copied to the training folders (`images/train`, `labels/train`).
   - Otherwise, every 8th pair is sent to the validation folders (`images/val`, `labels/val`), and the rest go to training.
5. Prefixes filenames with the source folder name to avoid collisions when datasets share image names.
6. Updates `Verified/Final_dataset/dataset.yaml` under the `# Content:` section, listing per-source train/val counts and appending overall totals.

If destination files already exist, the script aborts to prevent silent overwrites.

## Usage

Run the script from the project root:

```bash
python merge_datasets.py
```

After completion, `Verified/Final_dataset` contains the merged train/validation splits, `dataset.yaml` records the per-source image counts, and the original datasets remain unchanged.

## Paths

- Data root: current working directory (CWD). The script expects `Verified/` under CWD.
- Script directory: contains the script and its `.md` description.
