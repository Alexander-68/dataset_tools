# Correct Keypoints Script

This script processes YOLO11-pose annotation files (`.txt`) located in the `labels` folder. It specifically targets objects with `class_ID=0` (person) and modifies their keypoints based on visibility flags.

## Usage

1.  Ensure you have your annotation files in a folder (default is `labels`) under the current working directory (CWD).
2.  Run the script:
    ```bash
    python correct_keypoints.py [--folder FOLDER] [--threshold THRESHOLD]
    ```
    -   `--folder`: Specify the folder containing label files (default: `labels`).
    -   `--threshold`: Set the visibility probability threshold (default: `0.4`).
3.  The label files in the specified folder will be modified in-place.

## Logic

For each person object (class 0) with 17 keypoints (shape [17, 3]):
-   **Visibility Correction:**
    -   If visibility is a float/probability between `0.0` and `1.0`:
        -   `>= threshold` (default 0.4) is changed to `2` (visible).
        -   `< threshold` (default 0.4) is changed to `0` (invalid).
    -   If visibility is already outside the `[0, 1]` range (e.g., `2`), it is kept as an integer.
-   **Coordinate Reset:** If the resulting visibility is `0`, its X and Y coordinates are reset to `0.0`.

## Paths
-   Data root: current working directory (CWD). Relative `--folder` paths resolve from CWD.
-   Script directory: contains the script and its `.md` description.

## Requirements
-   Python 3.x
-   `tqdm` (install via `pip install tqdm`)
